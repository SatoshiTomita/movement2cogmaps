import argparse
import os
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


def curriculum_plan(
    behaviours: list[str], resume_from_stage: str | None
) -> list[tuple[int, str]]:
    """カリキュラム学習の実行"""
    if resume_from_stage is None:
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
    print(f"[+] Loading config: {conf_path}")
    print(f"[+] Saving checkpoints as: models/params/{save_name}/...")
    config_node: DictConfig = load_config(conf_path)
    config_node.alpha = args.alpha
    config_node.seed = args.seed
    config_node.device = args.device
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
    stage_plan = curriculum_plan(behaviours, args.resume_from_stage)
    curriculum_name = "-".join(behaviours)
    resume_weights_loaded = False

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
            if args.resume_from_stage is not None:
                resume_index = CURRICULUM_STAGES.index(args.resume_from_stage) + 1
                resume_path = (
                    Path("models/params")
                    / save_name
                    / f"alpha:{args.alpha}"
                    / f"seed:{args.seed}"
                    / f"{resume_index:02d}_{args.resume_from_stage}"
                    / f"{world.__class__.__name__}.ckpt"
                )
                print(f"[+] Loading best weights from: {resume_path}")
                load_best_weights(world, resume_path)
                resume_weights_loaded = True
        elif obs_shape != expected_obs_shape or action_dim != expected_action_dim:
            raise ValueError(
                "All curriculum stages must have the same observation and action "
                f"dimensions; expected {expected_obs_shape}/{expected_action_dim}, "
                f"got {obs_shape}/{action_dim} for {behaviour}"
            )

        if args.resume_from_stage is not None and not resume_weights_loaded:
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
        })
        os.makedirs("wandb", exist_ok=True)
        logger = WandbLogger(
            name=(
                f"{stage_index:02d}_{behaviour}_{save_name}_"
                f"alpha={args.alpha}_seed={args.seed}"
            ),
            project="WM_mobile",
            save_dir="./wandb/",
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
        callbacks = [instantiate(callback_cfg) for callback_cfg in callback_cfgs]
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
        )
        logger.experiment.finish()
        print(f"[+] Best checkpoint: {best_checkpoint}")
        if stage_position < len(stage_plan):
            print("[+] Loading the best stage weights for the next stage")
            load_best_weights(world, best_checkpoint)


def train(model: pl.LightningModule, train_loader, val_loader,
          trainer_cfg: TrainerConfig, n_epochs: int, path: str, callbacks: list,
          logger: WandbLogger, monitor_key: str = "loss",
          early_stop: int = 0) -> Path:
    save_model = ModelCheckpoint(
        dirpath=path,
        filename=model.__class__.__name__,
        monitor=f"{model.__class__.__name__}/{monitor_key}/val",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
        save_weights_only=True,
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
        **asdict(trainer_cfg),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if not save_model.best_model_path:
        raise RuntimeError("training finished without saving a best checkpoint")
    return Path(save_model.best_model_path)


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

    cli_args = parser.parse_args()
    if cli_args.epochs is not None and cli_args.epochs < 1:
        parser.error("--epochs must be positive")
    if cli_args.train and cli_args.world:
        main(cli_args)
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
