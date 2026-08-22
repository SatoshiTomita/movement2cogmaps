import tempfile

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch_optimizer
import wandb
from utils.utils import get_optimizer
from typing_extensions import Self
from typing import Union, Tuple

class LightningModuleBase(pl.LightningModule):

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[torch.optim.Optimizer, ...]]:

        return get_optimizer(self.parameters(), **self.optimizer_cfg)

    def training_step(
        self,
        batch,  # noqa: ANN401
        **_: dict,
    ):
        """Rollout training step."""
        loss_dict, *_ = self._shared_step(batch)
        log_loss_dict = {f"{self.__class__.__name__}/{k}/train": v for k, v in loss_dict.items()} 
        self.log_dict(log_loss_dict, prog_bar=True, sync_dist=True)
        return loss_dict

    def validation_step(
        self,
        batch,  # noqa: ANN401
        _: int,
    ):
        """
        Rollout validation step.

        The prefix of the key in `._shared_step` is written 'val_' and logged.
        """
        with torch.no_grad():
            loss_dict, pred_dict = self._shared_step(batch)
        loss_dict = {f"{self.__class__.__name__}/{k}/val": v for k, v in loss_dict.items()}
        self.log_dict(loss_dict, prog_bar=True, sync_dist=True)
        if pred_dict is not None:
            return dict(**loss_dict, **pred_dict)
        else:
            return loss_dict

    def _shared_step(
        self,
        batch,  # noqa: ANN401
    ):
        """
        Rollout common step for training and validation.

        It should return `dict` of the name of the loss and its value.
        """
        raise NotImplementedError

    @classmethod
    def load_from_wandb(cls, reference: str) -> Self:
        """Load the model from wandb checkpoint."""
        run = wandb.Api().run(reference)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_name, cpu = "best_model.ckpt", torch.device("cpu")
            ckpt = run.file(ckpt_name).download(replace=True, root=tmpdir)
            return cls.load_from_checkpoint(ckpt.name, map_location=cpu)
