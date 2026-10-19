import argparse
import os
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from architectures.world import CoarseWorldModel, WorldModel
from utils.callbacks import SwitchOptimizer
from utils.config import (
    CRSSMConfig,
    CRSSMV4Config,
    ExperimentConfig,
    TrainerConfig,
    load_config,
)
from utils.trainer import RNNTrainer
from utils.utils import torch_fix_seed


warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.*")
warnings.filterwarnings("ignore", ".*Input tensor has dimensions outside of.*")
warnings.filterwarnings("ignore", ".*torch.load.*")
os.environ["MUJOCO_GL"] = "egl"

CURRICULUM_STAGES = ("crawl", "walk", "run", "adult")


class CapacityViolationError(RuntimeError):
    """Raised when a stage remains outside its configured KL-capacity band."""


class CurriculumHealthMonitor(pl.Callback):
    """Log short- and long-horizon curriculum-learning health signals."""

    def __init__(
        self,
        model_name: str,
        stage_index: int,
        free_nats: float,
        kl_target: float | None = None,
        short_window: int = 5,
        long_window: int = 50,
        capacity_tolerance: float = 0.25,
        capacity_warmup_epochs: int = 50,
        capacity_patience: int = 10,
        adaptive_capacity: bool = False,
        capacity_adjustment_patience: int = 3,
        capacity_adjustment_factor: float = 2.0,
        max_kl_capacity_beta: float = 64.0,
    ) -> None:
        self.model_name = model_name
        self.stage_index = stage_index
        self.free_nats = free_nats
        self.kl_target = kl_target
        self.short_window = short_window
        self.long_window = long_window
        self.capacity_tolerance = capacity_tolerance
        self.capacity_warmup_epochs = capacity_warmup_epochs
        self.capacity_patience = capacity_patience
        self.adaptive_capacity = adaptive_capacity
        self.capacity_adjustment_patience = capacity_adjustment_patience
        self.capacity_adjustment_factor = capacity_adjustment_factor
        self.max_kl_capacity_beta = max_kl_capacity_beta
        self.capacity_violation_epochs = 0
        self.capacity_adjustments = 0
        self.stopped_for_capacity_violation = False
        self.reconstruction_history: list[float] = []
        self.kl_history: list[float] = []

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> float | None:
        value = metrics.get(key)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _long_horizon_metrics(self) -> dict[str, float]:
        if len(self.kl_history) < self.long_window * 2:
            return {}

        previous_kl_values = self.kl_history[-2 * self.long_window : -self.long_window]
        recent_kl_values = self.kl_history[-self.long_window :]
        previous_reconstruction = sum(
            self.reconstruction_history[-2 * self.long_window : -self.long_window]
        ) / self.long_window
        recent_reconstruction = (
            sum(self.reconstruction_history[-self.long_window :]) / self.long_window
        )
        previous_kl = sum(previous_kl_values) / self.long_window
        recent_kl = sum(recent_kl_values) / self.long_window
        kl_relative_change = abs(recent_kl - previous_kl) / max(abs(previous_kl), 1e-8)
        kl_relative_range = (max(recent_kl_values) - min(recent_kl_values)) / max(
            abs(recent_kl), 1e-8
        )
        reconstruction_change = recent_reconstruction - previous_reconstruction
        recent_window_starts = range(
            0, len(self.reconstruction_history) - self.long_window + 1
        )
        best_reconstruction_window = min(
            sum(self.reconstruction_history[start : start + self.long_window])
            / self.long_window
            for start in recent_window_starts
        )
        initial_kl_rose = True
        if self.stage_index == 1 and len(self.kl_history) >= 10:
            initial_kl_rose = (
                max(self.kl_history[:10])
                >= self.kl_history[0] * 2 + 0.1
            )
        kl_stable = kl_relative_change <= 0.05 and kl_relative_range <= 0.2
        reconstruction_converged = abs(reconstruction_change) <= 0.01
        noncollapsed_kl = recent_kl >= self.free_nats * 2
        near_best_reconstruction = recent_reconstruction <= best_reconstruction_window + 0.01
        expected_pattern = (
            initial_kl_rose
            and kl_stable
            and reconstruction_converged
            and noncollapsed_kl
            and near_best_reconstruction
        )
        return {
            "curriculum_health/long_kl_relative_change": kl_relative_change,
            "curriculum_health/long_kl_relative_range": kl_relative_range,
            "curriculum_health/long_reconstruction_change": reconstruction_change,
            "curriculum_health/long_kl_stable": float(kl_stable),
            "curriculum_health/reconstruction_converged": float(
                reconstruction_converged
            ),
            "curriculum_health/kl_not_collapsed": float(noncollapsed_kl),
            "curriculum_health/reconstruction_near_best": float(
                near_best_reconstruction
            ),
            "curriculum_health/expected_pattern_long": float(expected_pattern),
        }

    def _capacity_metrics(self, total_kl: float, epoch: int) -> dict[str, float]:
        if self.kl_target is None:
            return {}

        total_target = self.kl_target * 2
        relative_error = abs(total_kl - total_target) / total_target
        on_target = relative_error <= self.capacity_tolerance
        if epoch >= self.capacity_warmup_epochs:
            self.capacity_violation_epochs = (
                0 if on_target else self.capacity_violation_epochs + 1
            )
        stopped = self.capacity_violation_epochs >= self.capacity_patience
        return {
            "curriculum_health/capacity_target_total_kl": total_target,
            "curriculum_health/capacity_signed_error": total_kl - total_target,
            "curriculum_health/capacity_relative_error": relative_error,
            "curriculum_health/capacity_on_target": float(on_target),
            "curriculum_health/capacity_violation_epochs": float(
                self.capacity_violation_epochs
            ),
            "curriculum_health/stopped_for_capacity_violation": float(stopped),
        }

    def _adapt_capacity(
        self, pl_module: pl.LightningModule, capacity_metrics: dict[str, float]
    ) -> dict[str, float]:
        current_beta = float(pl_module.loss_cfg.get("kl_capacity_beta", 1.0))
        metrics = {
            "curriculum_health/kl_capacity_beta": current_beta,
            "curriculum_health/capacity_adjusted": 0.0,
            "curriculum_health/capacity_adjustments": float(self.capacity_adjustments),
        }
        if (
            not self.adaptive_capacity
            or self.kl_target is None
            or capacity_metrics["curriculum_health/capacity_on_target"]
            or self.capacity_violation_epochs < self.capacity_adjustment_patience
            or current_beta >= self.max_kl_capacity_beta
        ):
            return metrics

        next_beta = min(
            current_beta * self.capacity_adjustment_factor, self.max_kl_capacity_beta
        )
        OmegaConf.update(
            pl_module.loss_cfg, "kl_capacity_beta", next_beta, force_add=True
        )
        self.capacity_violation_epochs = 0
        self.capacity_adjustments += 1
        capacity_metrics["curriculum_health/capacity_violation_epochs"] = 0.0
        capacity_metrics["curriculum_health/stopped_for_capacity_violation"] = 0.0
        metrics.update(
            {
                "curriculum_health/kl_capacity_beta": next_beta,
                "curriculum_health/capacity_adjusted": 1.0,
                "curriculum_health/capacity_adjustments": float(
                    self.capacity_adjustments
                ),
            }
        )
        print(
            "[+] Increased KL capacity beta "
            f"from {current_beta:.6g} to {next_beta:.6g} after capacity drift"
        )
        return metrics

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        reconstruction = self._metric_value(
            metrics, f"{self.model_name}/reconstruction/val"
        )
        total_kl = self._metric_value(metrics, f"{self.model_name}/total_kl/val")
        if reconstruction is None or total_kl is None:
            return

        self.reconstruction_history.append(reconstruction)
        self.kl_history.append(total_kl)
        health_metrics = {
            "curriculum_health/reconstruction": reconstruction,
            "curriculum_health/total_kl": total_kl,
        }
        if len(self.kl_history) >= self.short_window * 2:
            previous_kl = (
                sum(
                    self.kl_history[-2 * self.short_window : -self.short_window]
                )
                / self.short_window
            )
            recent_kl = sum(self.kl_history[-self.short_window :]) / self.short_window
            recent_reconstruction = (
                sum(self.reconstruction_history[-self.short_window :])
                / self.short_window
            )
            initial_reconstruction = (
                sum(self.reconstruction_history[: self.short_window])
                / self.short_window
            )
            relative_kl_change = abs(recent_kl - previous_kl) / max(
                abs(previous_kl), 1e-8
            )
            initial_kl_rose = (
                max(self.kl_history[1 : self.short_window + 1])
                > self.kl_history[0]
            )
            reconstruction_improved = recent_reconstruction < initial_reconstruction
            kl_plateaued = relative_kl_change <= 0.05
            health_metrics.update(
                {
                    "curriculum_health/kl_recent_relative_change": relative_kl_change,
                    "curriculum_health/kl_rose_from_start": float(initial_kl_rose),
                    "curriculum_health/kl_plateaued": float(kl_plateaued),
                    "curriculum_health/reconstruction_improved": float(
                        reconstruction_improved
                    ),
                    "curriculum_health/expected_pattern": float(
                        initial_kl_rose and kl_plateaued and reconstruction_improved
                    ),
                }
            )
        health_metrics.update(self._long_horizon_metrics())
        capacity_metrics = self._capacity_metrics(total_kl, trainer.current_epoch)
        health_metrics.update(capacity_metrics)
        health_metrics.update(self._adapt_capacity(pl_module, capacity_metrics))
        if capacity_metrics.get("curriculum_health/stopped_for_capacity_violation"):
            self.stopped_for_capacity_violation = True
            trainer.should_stop = True
            print(
                "[+] Stopping stage because KL remained outside the capacity "
                f"target for {self.capacity_violation_epochs} validation epochs"
            )

        trainer.logger.log_metrics(health_metrics, step=trainer.global_step)
        print(
            "[+] Curriculum health "
            f"epoch={trainer.current_epoch} reconstruction={reconstruction:.6f} "
            f"total_kl={total_kl:.6f}"
        )


class WorldModelCollate:
    """既存のDataloaderの6要素出力をWorldModelに入力する3要素へ変換する
    scene, velocity, rot_velocity, position, theta, labels
                         ↓
    action(移動速度+回転速度), observation(現在画像), target(1ステップ後の画像)
    """

    def __init__(self, obs_shape, action_transform):
        self.obs_shape = tuple(obs_shape)
        self.action_transform = action_transform

    def __call__(self, batch):
        if len(batch) != 1:
            raise ValueError("The current recurrent dataloader expects batch_size=1")
        scene, velocity, rot_velocity, _, _, labels = batch[0]

        velocity = velocity[:, 0]
        rot_velocity = rot_velocity[:, 0]
        target = labels[:, 0]

        action = torch.cat([velocity, rot_velocity], dim=-1)
        action = self.action_transform(action)

        n_trajectories, n_steps = scene.shape[:2]
        scene = scene.reshape(n_trajectories, n_steps, *self.obs_shape)
        target = target.reshape(n_trajectories, n_steps, *self.obs_shape)

        # action[T,B,F](並進速度＋回転速度)、scene(現在観測)[T,B,C,H,W]、target(1ステップ後観測)[T,B,C,H,W]の順に返す
        return (
            action.transpose(0, 1),
            scene.transpose(0, 1),
            target.transpose(0, 1),
        )


def load_best_weights(model: pl.LightningModule, checkpoint_path: Path) -> None:
    """保存した重みをロードする"""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"best checkpoint for curriculum resume not found: {checkpoint_path}"
        )
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = saved.get("state_dict", saved)
    # These buffers are regenerated from the current configuration and older
    # checkpoints may contain versions with incompatible shapes.
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if key != "action_transform.k" and "codebook.mask" not in key
    }
    model.load_state_dict(state_dict, strict=True)


def is_full_training_checkpoint(checkpoint_path: Path) -> bool:
    """Return whether a checkpoint can restore optimizer and scheduler state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return "optimizer_states" in checkpoint and "lr_schedulers" in checkpoint


def curriculum_plan(
    behaviours: list[str],
    resume_from_stage: str | None,
    continue_stage: str | None,
) -> list[tuple[int, str]]:
    """Build a valid ordered curriculum plan."""
    if continue_stage is not None:
        if behaviours != [continue_stage]:
            raise ValueError(
                f"A continued stage must train only {continue_stage!r}; got "
                f"{','.join(behaviours)}"
            )
        return [(CURRICULUM_STAGES.index(continue_stage) + 1, continue_stage)]

    if resume_from_stage is None:
        expected = list(CURRICULUM_STAGES[: len(behaviours)])
        if len(behaviours) > 1 and behaviours != expected:
            raise ValueError(
                "A multi-stage curriculum must follow crawl,walk,run,adult; got "
                f"{','.join(behaviours)}"
            )
        return list(enumerate(behaviours, start=1))

    resume_index = CURRICULUM_STAGES.index(resume_from_stage)
    expected = list(
        CURRICULUM_STAGES[resume_index + 1 : resume_index + 1 + len(behaviours)]
    )
    if behaviours != expected:
        expected_text = ",".join(expected) or "(no later stages)"
        raise ValueError(
            f"A curriculum resumed from {resume_from_stage!r} must continue with "
            f"{expected_text}; got {','.join(behaviours)}"
        )
    return [
        (CURRICULUM_STAGES.index(behaviour) + 1, behaviour)
        for behaviour in behaviours
    ]


def main(args):
    conf_path = f"models/cfg/{args.model_name}.yaml"
    save_name = getattr(args, "save_name", None) or args.model_name
    resume_save_name = args.resume_from_save_name or save_name
    print(f"[+] Loading config: {conf_path}")
    print(f"[+] Saving checkpoints as: models/params/{save_name}/...")
    config_node: DictConfig = load_config(conf_path)
    config_node.alpha = args.alpha
    config_node.seed = args.seed
    config_node.device = args.device
    if args.kl_target is not None:
        OmegaConf.update(
            config_node.world.loss_cfg,
            "kl_target",
            args.kl_target,
            force_add=True,
        )
        OmegaConf.update(
            config_node.world.loss_cfg,
            "kl_capacity_beta",
            args.kl_capacity_beta,
            force_add=True,
        )
        OmegaConf.update(
            config_node.world.loss_cfg,
            "kl_capacity_mode",
            args.kl_capacity_mode,
            force_add=True,
        )
        config_node.world.loss_cfg.free_nats = 0.0
    if args.epochs is not None:
        config_node.epochs.world = args.epochs
    if args.disable_early_stop:
        config_node.world.loss_cfg.early_stop = 0
    torch_fix_seed(config_node.seed)

    # CLI values override the data settings in the model configuration.
    data_cfg = config_node.datamodule
    args.data_root = args.data_root or data_cfg.data_root
    args.behaviour = args.behaviour or data_cfg.behaviour
    args.env = args.env or data_cfg.env
    args.env_dim = args.env_dim if args.env_dim is not None else data_cfg.env_dim
    args.stride = args.stride if args.stride is not None else data_cfg.stride
    args.bptt_steps = (
        args.bptt_steps if args.bptt_steps is not None else data_cfg.chunk_size
    )
    args.num_workers = (
        args.num_workers if args.num_workers is not None else data_cfg.num_workers
    )

    behaviours = args.curriculum or [args.behaviour]
    valid_behaviours = set(CURRICULUM_STAGES)
    unknown = [behaviour for behaviour in behaviours if behaviour not in valid_behaviours]
    if unknown:
        raise ValueError(f"Unknown curriculum behaviours: {unknown}")

    callback_cfgs = config_node.pop("callbacks", [])
    world = None
    config = None
    expected_obs_shape = None
    expected_action_dim = None
    stage_plan = curriculum_plan(
        behaviours, args.resume_from_stage, args.continue_stage
    )
    curriculum_name = "-".join(behaviours)
    resume_weights_loaded = False
    source_stage = args.continue_stage or args.resume_from_stage
    training_resume_checkpoint: Path | None = None

    for stage_position, (stage_index, behaviour) in enumerate(stage_plan, start=1):
        print(
            f"\n[+] Curriculum stage {stage_position}/{len(behaviours)} "
            f"(canonical stage {stage_index}): "
            f"{behaviour}"
        )
        loader_args = SimpleNamespace(
            architecture="crssmv4",
            behaviour=behaviour,
            env=args.env,
            env_dim=args.env_dim,
            moredata=None,
            activity_only=False,
            # RNNTrainer calculates scheduler defaults during initialization.
            epochs=2,
            stride=args.stride,
            bptt_steps=args.bptt_steps,
            n_future_pred=1,
            n_gridcells=0,
            num_workers=args.num_workers,
            name_prefix=None,
            pretrained_model_folder=None,
            reset_hidden_at=None,
            dropouts=[0, 0, 0],
            latent_dim=1,
            stoch_dim=1,
            kl_scale=1.0,
            free_nats=0.0,
            hidden_reg=0.0,
            weights_reg=0.0,
            seed=args.seed,
            stoch_dist="normal",
        )
        data_loader_factory = RNNTrainer(
            loader_args,
            args.data_root,
            torch.device("cpu"),
            create_dir=False,
        )
        # シミュレーション結果を取得
        videos, thetas, positions, velocities, rot_velocities = (
            data_loader_factory.load_simulations()
        )
        # 0,1に正規化
        videos, velocities, rot_velocities = data_loader_factory.preprocess_data(
            videos, velocities, rot_velocities
        )
        # train,validationで分割
        split = data_loader_factory.combine_videos(
            videos, velocities, rot_velocities, positions, thetas
        )
        train_arrays = split[:5]
        val_arrays = split[5:10]

        frame_height = loader_args.frame_dim[1] // loader_args.frame_subsampling
        frame_width = loader_args.frame_dim[0] // loader_args.frame_subsampling
        obs_shape = (1, frame_height, frame_width)
        action_dim = train_arrays[1].shape[-1] + train_arrays[2].shape[-1]

        if world is None:
            config_node.obs_shape = obs_shape
            config_node.action_dim = action_dim
            config_node.datamodule.behaviour = behaviour
            OmegaConf.resolve(config_node)
            config = instantiate(config_node)
            config.world.dict2dc()
            if isinstance(config.world.dynamics_cfg, (CRSSMV4Config, CRSSMConfig)):
                world = CoarseWorldModel(config.world)
            else:
                world = WorldModel(config.world)
            expected_obs_shape = obs_shape
            expected_action_dim = action_dim
            if source_stage is not None:
                resume_index = CURRICULUM_STAGES.index(source_stage) + 1
                resume_path = (
                    Path("models/params")
                    / resume_save_name
                    / f"alpha:{args.alpha}"
                    / f"seed:{args.seed}"
                    / f"{resume_index:02d}_{source_stage}"
                    / f"{world.__class__.__name__}.ckpt"
                )
                print(f"[+] Loading best weights from: {resume_path}")
                if args.continue_stage is not None:
                    if not is_full_training_checkpoint(resume_path):
                        raise ValueError(
                            "Cannot continue a weights-only checkpoint. Start a "
                            "fresh stage with full checkpoint saving, then resume "
                            "from that checkpoint."
                        )
                    training_resume_checkpoint = resume_path
                else:
                    load_best_weights(world, resume_path)
                resume_weights_loaded = True
        elif obs_shape != expected_obs_shape or action_dim != expected_action_dim:
            raise ValueError(
                "All curriculum stages must have the same observation and action "
                f"dimensions; expected {expected_obs_shape}/{expected_action_dim}, "
                f"got {obs_shape}/{action_dim} for {behaviour}"
            )

        if source_stage is not None and not resume_weights_loaded:
            raise RuntimeError("resume checkpoint weights were not loaded")

        collate_fn = WorldModelCollate(obs_shape, world.action_transform)
        train_loader = data_loader_factory.generate_dataloader(
            *train_arrays, collate_fn=collate_fn
        )
        val_loader = data_loader_factory.generate_dataloader(
            *val_arrays, collate_fn=collate_fn
        )
        base_path = (
            f"models/params/{save_name}/alpha:{args.alpha}/seed:{args.seed}"
        )
        path = os.path.join(base_path, f"{stage_index:02d}_{behaviour}")
        os.makedirs(path, exist_ok=True)
        if args.clean_old_checkpoints:
            stage_path = Path(path)
            legacy_checkpoints = [
                *stage_path.glob("ep:*.pth"),
                *stage_path.glob("last*.ckpt"),
                *stage_path.glob("*-v*.ckpt"),
            ]
            for legacy_checkpoint in sorted(set(legacy_checkpoints)):
                legacy_checkpoint.unlink()
                print(f"[+] Removed legacy checkpoint: {legacy_checkpoint}")

        logger_config = asdict(config)
        logger_config["datamodule"]["behaviour"] = behaviour
        logger_config.update({
            "config_name": args.model_name,
            "save_name": save_name,
            "behaviour": behaviour,
            "curriculum": behaviours,
            "curriculum_stage": stage_index,
            "monitoring": {
                "reconstruction": "CoarseWorldModel/reconstruction/{train,val}",
                "precise_kl": "CoarseWorldModel/precise_kl/{train,val}",
                "coarse_kl": "CoarseWorldModel/coarse_kl/{train,val}",
                "total_kl": "CoarseWorldModel/total_kl/{train,val}",
                "selection_metric": f"{world.__class__.__name__}/{config.world.loss_cfg.get('monitor_key', 'loss')}/val",
            },
        })
        os.makedirs("wandb", exist_ok=True)
        logger = WandbLogger(
            name=(
                f"{stage_index:02d}_{behaviour}_{save_name}_"
                f"alpha={args.alpha}_seed={args.seed}"
            ),
            project="WM_mobile",
            save_dir=".",
            config=logger_config,
            tags=[
                f"alpha={args.alpha}",
                f"seed_{args.seed}",
                behaviour,
                args.env,
                f"curriculum={curriculum_name}",
            ],
            group=f"{save_name}_{curriculum_name}_seed={args.seed}",
        )
        for metric in (
            "reconstruction",
            "precise_kl",
            "coarse_kl",
            "total_kl",
            "loss",
        ):
            logger.experiment.define_metric(
                f"{world.__class__.__name__}/{metric}/train", step_metric="epoch"
            )
            logger.experiment.define_metric(
                f"{world.__class__.__name__}/{metric}/val", step_metric="epoch"
            )
        callbacks = [instantiate(callback_cfg) for callback_cfg in callback_cfgs]
        callbacks.append(
            CurriculumHealthMonitor(
                world.__class__.__name__,
                stage_index=stage_index,
                free_nats=float(config.world.loss_cfg.free_nats),
                kl_target=config.world.loss_cfg.get("kl_target"),
                adaptive_capacity=args.adaptive_kl_capacity,
                capacity_adjustment_patience=args.capacity_adjustment_patience,
                capacity_adjustment_factor=args.capacity_adjustment_factor,
                max_kl_capacity_beta=args.max_kl_capacity_beta,
            )
        )
        try:
            best_checkpoint = train(
                model=world,
                train_loader=train_loader,
                val_loader=val_loader,
                trainer_cfg=config.trainer,
                n_epochs=config.epochs.world,
                path=path,
                callbacks=callbacks,
                logger=logger,
                monitor_key=config.world.loss_cfg.get("monitor_key", "loss"),
                early_stop=config.world.loss_cfg.get("early_stop", 0),
                limit_train_batches=args.limit_train_batches,
                limit_val_batches=args.limit_val_batches,
                resume_checkpoint=training_resume_checkpoint,
            )
        finally:
            logger.experiment.finish()
        print(f"[+] Best checkpoint: {best_checkpoint}")
        if stage_position < len(stage_plan):
            print("[+] Loading the best stage weights for the next stage")
            load_best_weights(world, best_checkpoint)


def train(model: pl.LightningModule, train_loader, val_loader,
          trainer_cfg: TrainerConfig, n_epochs: int, path: str, callbacks: list,
          logger: WandbLogger, monitor_key: str = "loss",
          early_stop: int = 0, limit_train_batches: float = 1.0,
          limit_val_batches: float = 1.0,
          resume_checkpoint: Path | None = None) -> Path:
    save_model = ModelCheckpoint(
        dirpath=path,
        filename=model.__class__.__name__,
        monitor=f"{model.__class__.__name__}/{monitor_key}/val",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
        save_weights_only=False,
        save_on_train_epoch_end=False,
        save_last=False,
    )
    callbacks.extend([
        SwitchOptimizer(),
        save_model,
    ])
    if early_stop > 0:
        callbacks.append(EarlyStopping(
            monitor=f"{model.__class__.__name__}/{monitor_key}/val",
            patience=early_stop,
            mode="min",
        ))
    trainer = pl.Trainer(
        logger=logger,
        fast_dev_run=False,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=n_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        **asdict(trainer_cfg),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
    )
    capacity_stop = any(
        isinstance(callback, CurriculumHealthMonitor)
        and callback.stopped_for_capacity_violation
        for callback in callbacks
    )
    if capacity_stop:
        for checkpoint in Path(path).glob("*.ckpt"):
            checkpoint.unlink()
            print(f"[+] Removed failed-stage checkpoint: {checkpoint}")
        raise CapacityViolationError(
            "stage stopped because KL remained outside the capacity target"
        )
    if not save_model.best_model_path:
        raise RuntimeError("training finished without saving a best checkpoint")
    best_checkpoint = Path(save_model.best_model_path).resolve()
    for checkpoint in Path(path).glob("*.ckpt"):
        if checkpoint.resolve() != best_checkpoint:
            checkpoint.unlink()
            print(f"[+] Removed non-best checkpoint: {checkpoint}")
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="maincrssmV4.py")
    parser.add_argument("--model_name", default="0824base")
    parser.add_argument(
        "--save-name",
        "--save_name",
        default=None,
        help=(
            "Checkpoint/W&B run name. Defaults to --model_name, which remains "
            "the configuration filename."
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--kl-target",
        type=float,
        default=None,
        help=(
            "Use a KL-capacity objective with this target per precise/coarse "
            "latent level; this disables the free-nats floor."
        ),
    )
    parser.add_argument(
        "--kl-capacity-beta",
        type=float,
        default=1.0,
        help="Strength of the KL-capacity objective when --kl-target is set.",
    )
    parser.add_argument(
        "--kl-capacity-mode",
        choices=("absolute", "quadratic"),
        default="absolute",
        help=(
            "Absolute or normalized quadratic penalty around --kl-target; "
            "quadratic increases corrective pressure as KL drifts farther."
        ),
    )
    parser.add_argument(
        "--adaptive-kl-capacity",
        action="store_true",
        help="Increase the KL-capacity coefficient after persistent target drift.",
    )
    parser.add_argument(
        "--capacity-adjustment-patience",
        type=int,
        default=3,
        help="Out-of-band validation epochs before increasing the KL coefficient.",
    )
    parser.add_argument(
        "--capacity-adjustment-factor",
        type=float,
        default=2.0,
        help="Multiplier used by --adaptive-kl-capacity.",
    )
    parser.add_argument(
        "--max-kl-capacity-beta",
        type=float,
        default=64.0,
        help="Maximum adaptive KL-capacity coefficient.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", "-tr", action="store_true")
    parser.add_argument("--world", "-w", action="store_true")
    parser.add_argument("--evaluate", "-e", action="store_true")
    parser.add_argument("--load-last", "-l", action="store_true")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--imagine", type=int, default=0)
    parser.add_argument("--interval", type=int, default=None)

    parser.add_argument("--data-root", default=None)
    parser.add_argument("--behaviour", default=None)
    parser.add_argument(
        "--curriculum",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=None,
        help="Comma-separated stages, e.g. crawl,walk,run,adult",
    )
    parser.add_argument(
        "--resume-from-stage",
        choices=CURRICULUM_STAGES,
        default=None,
        help=(
            "Load the best checkpoint from this completed stage, then train the "
            "subsequent stages specified by --curriculum. For example: "
            "--resume-from-stage crawl --curriculum walk,run,adult"
        ),
    )
    parser.add_argument(
        "--resume-from-save-name",
        default=None,
        help=(
            "Experiment save name containing --resume-from-stage's best "
            "checkpoint; defaults to --save-name."
        ),
    )
    parser.add_argument(
        "--continue-stage",
        choices=CURRICULUM_STAGES,
        default=None,
        help=(
            "Load the best checkpoint from this stage and continue training "
            "that exact stage. Requires --curriculum with the same stage."
        ),
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--env-dim", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--bptt-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs for each curriculum stage (overrides the config).",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="Run every requested epoch instead of stopping on a validation plateau.",
    )
    parser.add_argument(
        "--clean-old-checkpoints",
        action="store_true",
        help=(
            "Remove legacy periodic/last checkpoints in each target stage before "
            "training; the existing best checkpoint is overwritten only after "
            "validation."
        ),
    )
    parser.add_argument(
        "--limit-train-batches",
        type=float,
        default=1.0,
        help="Fraction of training batches to run per epoch; intended for smoke tests.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=float,
        default=1.0,
        help="Fraction of validation batches to run per epoch; intended for smoke tests.",
    )

    cli_args = parser.parse_args()
    if cli_args.epochs is not None and cli_args.epochs < 1:
        parser.error("--epochs must be positive")
    if cli_args.kl_target is not None and cli_args.kl_target <= 0:
        parser.error("--kl-target must be positive")
    if cli_args.resume_from_stage is not None and cli_args.continue_stage is not None:
        parser.error("--resume-from-stage and --continue-stage are exclusive")
    if cli_args.resume_from_save_name is not None and (
        cli_args.resume_from_stage is None and cli_args.continue_stage is None
    ):
        parser.error(
            "--resume-from-save-name requires --resume-from-stage or --continue-stage"
        )
    if cli_args.kl_capacity_beta <= 0:
        parser.error("--kl-capacity-beta must be positive")
    if cli_args.capacity_adjustment_patience < 1:
        parser.error("--capacity-adjustment-patience must be positive")
    if cli_args.capacity_adjustment_factor <= 1:
        parser.error("--capacity-adjustment-factor must be greater than one")
    if cli_args.max_kl_capacity_beta < cli_args.kl_capacity_beta:
        parser.error("--max-kl-capacity-beta must be at least --kl-capacity-beta")
    if not 0 < cli_args.limit_train_batches <= 1:
        parser.error("--limit-train-batches must be in (0, 1]")
    if not 0 < cli_args.limit_val_batches <= 1:
        parser.error("--limit-val-batches must be in (0, 1]")
    if cli_args.train and cli_args.world:
        try:
            main(cli_args)
        except CapacityViolationError as error:
            print(f"[+] Capacity-controlled stop: {error}", file=sys.stderr)
            raise SystemExit(20) from error
    if cli_args.evaluate and cli_args.world:
        from utils.eval import eval_world

        eval_world(
            cli_args.model_name,
            cli_args.alpha,
            cli_args.seed,
            cli_args.device,
            cli_args.load_last,
            indices=[cli_args.idx] if cli_args.idx is not None else None,
            imagine=cli_args.imagine,
            interval=cli_args.interval,
        )
