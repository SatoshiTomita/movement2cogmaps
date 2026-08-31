import pytorch_lightning as pl
import numpy as np
import torch
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
import wandb
from typing import TYPE_CHECKING, Union
from einops import rearrange
from schedulefree import RAdamScheduleFree

if TYPE_CHECKING:
    from src.data.world_dataset import WorldDataset


class SaveParams(pl.Callback):
    def __init__(self, path: str, save_every_n_epoch: int):
        super().__init__()
        self.path = path
        self.save_every_n_epoch = save_every_n_epoch

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.current_epoch % self.save_every_n_epoch == 0:
            state_dict = pl_module.state_dict()
            torch.save(
                state_dict, f"{self.path}/ep:{trainer.current_epoch}.pth")


class ProgressBarCallback(RichProgressBar):
    """
    Make the progress bar richer.

    References
    ----------
    * https://qiita.com/akihironitta/items/edfd6b29dfb67b17fb00
    """

    def __init__(self) -> None:
        """Rich progress bar with custom theme."""
        theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        super().__init__(theme=theme)

class LogOriginalImage(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dataset: "WorldDataset" = trainer.datamodule.val_data
        image_inputs = dataset.image_inputs
        print(image_inputs.shape)

        original_img = image_inputs.transpose(0, 1).detach().numpy()

        trainer.logger.experiment.log(
            {"original": [wandb.Video(image.astype(np.uint8), fps=15, format="mp4") for image in original_img]}
        )

class VisualizeReconstruction(pl.Callback):
    def __init__(self, save_every_n_epoch: int, names: list = ["prediction"]):
        super().__init__()
        self.save_every_n_epoch = save_every_n_epoch
        self.names = names

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # This visualization callback belongs to the former src.data pipeline.
        # Import it only when that optional callback is actually used.
        from src.data.make_predata import unscale_obs

        if trainer.current_epoch % self.save_every_n_epoch != 0:
            return
        media_dict = {}
        for name in self.names:
            reconstruction = outputs[name].cpu().detach()
            if reconstruction.shape[0] > reconstruction.shape[1]:
                reconstruction = reconstruction.transpose(0, 1)
    
                
            reconstruction = unscale_obs(reconstruction)

            media_dict[name] = [
                wandb.Video(image.astype(np.uint8), fps=15, format="mp4") for image in reconstruction
            ]

        trainer.logger.experiment.log(media_dict)

class Unfreeze(pl.Callback):
    def __init__(self, epoch: int, module_name: str = "encoder"):
        super().__init__()
        self.epoch = epoch
        self.module_name = module_name

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        if trainer.current_epoch == self.epoch:
            getattr(pl_module, self.module_name).unfreeze()

class SwitchOptimizer(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer,
        opt_idx: int = 0,
        ):

        optimizer = pl_module.optimizers()
        if isinstance(optimizer, RAdamScheduleFree):
            optimizer.train()

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, RAdamScheduleFree):
            optimizer.train()

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, RAdamScheduleFree):
            optimizer.eval()

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        optimizer = pl_module.optimizers()
        if isinstance(optimizer, RAdamScheduleFree):
            optimizer.eval()

      
