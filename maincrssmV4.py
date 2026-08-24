import argparse
import os
import warnings
import torch
from dataclasses import asdict

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from src.data.world_dataset import DatasetModule, split_id
from src.data.make_predata import prepare_dataset
from src.modules.world import (WorldModel, CoarseWorldModel )
from src.utils.callbacks import SaveParams, SwitchOptimizer
from src.utils.config import (AgentConfig, ExperimentConfig, 
                              load_config, TrainerConfig, 
                              CRSSMConfig, CRSSMV4Config, 
                              )
from src.utils.eval import eval_world
from src.utils.utils import torch_fix_seed


warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*exists and is not empty.")
warnings.filterwarnings("ignore", ".*Input tensor has dimensions outside of.*")
warnings.filterwarnings("ignore", ".*torch.load.*")
warnings.filterwarnings(
    "ignore", ".*I found a path object that I don't think is part of a bar chart.*"
)
warnings.filterwarnings("ignore", ".*Tight.*")
os.environ["MUJOCO_GL"] = "egl"


def main(
    model_name: str,
    alpha: float,
    seed: int,
    device: int,
    train_world: bool,
):

    conf_path = f"models/cfg/{model_name}.yaml"

    config: DictConfig = load_config(conf_path)
    config.alpha = alpha
    config.seed = seed
    config.device = device

    torch_fix_seed(config.seed)

    indices = split_id(
            config.idx_splitter.num, 
            config.idx_splitter.change_point, 
            config.idx_splitter.n_val_each
            )
    # 修正点3:prepare_dataset(config.data_dir)をコメントアウト
    # prepare_dataset(config.data_dir)
    data_cfg = load_config(f"data/{config.data_dir}/config.yaml")

    env_cfg = dict(
        obs_shape=data_cfg.obs_shape,
        action_dim=data_cfg.action_dim,
    )


    print("envrionment:", env_cfg)
    config.update(env_cfg)
    OmegaConf.resolve(config)
    config: ExperimentConfig = instantiate(
        config,
    )

    callbacks = config.callbacks

    tags = [f"alpha={alpha}", f"seed_{config.seed}", f"{config.data_dir}"]

    os.makedirs("wandb", exist_ok=True)
    name = f"{config.data_dir}_{model_name}_alpha={config.alpha}_seed={config.seed}"
    config.tuple_callbacks()
    config.dc2dict()
    logger = WandbLogger(
        name=name,
        project="WM_mobile",
        save_dir="./wandb/",
        config=config,
        tags=tags,
    )

    if train_world:
        path = f"models/params/{model_name}/alpha:{config.alpha}/seed:{config.seed}/"
        os.makedirs(path, exist_ok=True)
        config.world.dict2dc()
        if isinstance(config.world.dynamics_cfg, CRSSMV4Config) or isinstance(config.world.dynamics_cfg, CRSSMConfig):
            world = CoarseWorldModel(config.world)
        else:
            world = WorldModel(config.world)
        datamodule = DatasetModule(
                config.batch_size.world,
                config.datamodule,
                "WorldDataset",
                indices,
                action_transform=world.action_transform,
                fine_tune=config.world.finetune_decoder
                )
        train(
            model=world,
            datamodule=datamodule,
            trainer_cfg=config.trainer,
            n_epochs=config.epochs.world,
            path=path,
            callbacks=callbacks,
            logger=logger,
            monitor_key=config.world.loss_cfg.get("monitor_key", 'loss'),
            early_stop=config.world.loss_cfg.get("early_stop", 0)
        )

def train(
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer_cfg: TrainerConfig,
        n_epochs: int,
        path: str,
        callbacks: list,
        logger: WandbLogger,
        monitor_key: str = "loss",
        early_stop: int = 0,
        ):


    save_model = ModelCheckpoint(
        dirpath=path,
        filename=model.__class__.__name__,
        monitor=f"{model.__class__.__name__}/{monitor_key}/val",
        save_weights_only=True,
        save_on_train_epoch_end=False,
        save_last=True,
    )
    save_params = SaveParams(path, save_every_n_epoch=n_epochs // 10)
    switch_optimizer = SwitchOptimizer()
    callbacks.append(switch_optimizer)

    callbacks.append(save_model)
    callbacks.append(save_params)
    if early_stop > 0:
        callbacks.append(
            EarlyStopping(
                monitor=f"{model.__class__.__name__}/{monitor_key}/val",
                patience=early_stop,
                mode="min",
            )
        )
    trainer = pl.Trainer(
        logger=logger,
        fast_dev_run=False,
        callbacks=callbacks,
        detect_anomaly=False,
        max_epochs=n_epochs,
        **asdict(trainer_cfg),
    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main.py", epilog="end", add_help=True)
    parser.add_argument("--model_name", type=str, help="model name in models/", default="default")
    parser.add_argument("--alpha", type=float, help="alpha setting", default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", "-tr", action="store_true", help="train model")
    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate")
    parser.add_argument("--robot_test", "-rt", action="store_true", help="robot test")
    parser.add_argument("--robot_trials", "-rtr", type=int, help="robot trials", default=0)
    parser.add_argument("--check", "-c", action="store_true", help="check")
    parser.add_argument("--world", "-w", action="store_true", help="world", default=False)
    parser.add_argument("--agent", "-a", action="store_true", help="agent", default=False)
    parser.add_argument("--load-last", "-l", action="store_true", help="load last model", default=False)
    parser.add_argument('--test_name', type=str, help='test name in tests/', default="")
    parser.add_argument('--feature', type=str, default=False)
    parser.add_argument('--length', type=int, default=50)
    parser.add_argument('--efe_freq', type=int, default=5)
    parser.add_argument('--preference', '-p', type=str, default="bL_lB_l2F")
    parser.add_argument('--pref_std', type=float, default=0.1)
    parser.add_argument('--n_sample', type=int, default=256)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--use-policy', action='store_true', help='use policy', default=False)
    parser.add_argument("--use-nce", action="store_true", help="use nce", default=False)
    parser.add_argument("--data-dir", type=str, help="data directory", default="")
    parser.add_argument("--idx", type=int, help="index of data", default=None)
    parser.add_argument("--stepwise", action="store_true", help="stepwise", default=False)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--use-essential", action="store_true", help="use essential", default=False)
    parser.add_argument("--red", action="store_true", help="use red", default=False)
    parser.add_argument("--blue", action="store_true", help="use blue", default=False)
    parser.add_argument("--imagine", type=int, default=0)
    parser.add_argument("--interval", type=int, default=None)

    args = parser.parse_args()
    if args.train:
        main(
            args.model_name,
            args.alpha,
            args.seed,
            args.device,
            args.world,
        )
    if args.evaluate or args.train:
        if args.world:
            eval_world(
                    args.model_name, 
                    args.alpha, 
                    args.seed, 
                    args.device, 
                    args.load_last,
                    indices=[args.idx] if args.idx is not None else None,
                    imagine=args.imagine,
                    interval=args.interval
                    )
