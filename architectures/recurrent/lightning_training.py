"""PyTorch Lightning utilities for the existing world-model classes."""

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from utils.utils import get_optimizer


class WorldModelBatchCollator:
    """Convert recurrent samples to transition-aligned WorldModel inputs.

    The dataset provides current observations ``[o_t, ..., o_(t+T-1)]`` and
    labels ``[o_(t+1), ..., o_(t+T)]``.  The current observations are the
    reconstruction targets.  Labels are used only to infer posterior states
    after each action; those states become the current states at the following
    time indices/BPTT windows.
    """

    def __init__(self, frame_shape):
        self.height, self.width = frame_shape

    def __call__(self, samples):
        if len(samples) != 1:
            raise ValueError("WorldModel Lightning loader requires batch_size=1")

        scene, velocity, rotational_velocity, _, _, labels = samples[0]
        expected = self.height * self.width
        if scene.shape[-1] != expected:
            raise ValueError(
                f"WorldModel expected {expected} image features, "
                f"got {scene.shape[-1]}"
            )

        action = torch.cat(
            [velocity[:, 0], rotational_velocity[:, 0]], dim=-1
        )
        current_observation = scene.reshape(
            *scene.shape[:-1], 1, self.height, self.width
        )
        post_action_observation = labels[:, 0].reshape(
            *labels[:, 0].shape[:-1], 1, self.height, self.width
        )

        return (
            action.transpose(0, 1).contiguous(),
            post_action_observation.transpose(0, 1).contiguous(),
            current_observation.transpose(0, 1).contiguous(),
            current_observation[:, 0].contiguous(),
        )


class LightningWorldModelRunner(pl.LightningModule):
    """Run truncated BPTT around an unchanged world-model implementation."""

    def __init__(self, model, bptt_steps):
        super().__init__()
        self.model = model
        self.bptt_steps = bptt_steps
        self.automatic_optimization = False

    def configure_optimizers(self):
        return get_optimizer(self.model.parameters(), **self.model.optimizer_cfg)

    def on_fit_start(self):
        """Guarantee that the wrapped model follows Lightning's device."""
        self.model.to(self.device)

    def _chunks(self, batch):
        if len(batch) == 3:
            action, observation, target = batch
            initial_obs = None
        elif len(batch) == 4:
            action, observation, target, initial_obs = batch
        else:
            raise ValueError(
                "WorldModel batches must have three or four tensors"
            )
        return (
            action.split(self.bptt_steps, dim=0),
            observation.split(self.bptt_steps, dim=0),
            target.split(self.bptt_steps, dim=0),
            initial_obs,
        )

    def _model_step(self, batch, initialize):
        loss_dict, embed_obs, prediction = self.model._train(
            batch, init=initialize
        )
        return loss_dict

    @staticmethod
    def _mean_logs(logs, count):
        return {key: value / count for key, value in logs.items()}

    def _add_logs(self, totals, loss_dict):
        for key, value in loss_dict.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device)
            totals[key] = totals.get(key, 0) + value.detach()

    def training_step(self, batch, batch_idx):
        action_chunks, obs_chunks, target_chunks, initial_obs = self._chunks(batch)
        optimizer = self.optimizers()
        totals = {}
        for index, transition_chunk in enumerate(
            zip(action_chunks, obs_chunks, target_chunks)
        ):
            optimizer.zero_grad()
            chunk = (*transition_chunk, initial_obs)
            loss_dict = self._model_step(chunk, initialize=index == 0)
            self.manual_backward(loss_dict["loss"])
            optimizer.step()
            self.model.dynamics.detach()
            self._add_logs(totals, loss_dict)

        totals = self._mean_logs(totals, len(action_chunks))
        self.log_dict(
            {f"{self.model.__class__.__name__}/{key}/train": value
             for key, value in totals.items()},
            on_step=False,
            on_epoch=True,
            prog_bar="loss" in totals,
        )
        return totals["loss"]

    def validation_step(self, batch, batch_idx):
        action_chunks, obs_chunks, target_chunks, initial_obs = self._chunks(batch)
        totals = {}
        for index, transition_chunk in enumerate(
            zip(action_chunks, obs_chunks, target_chunks)
        ):
            chunk = (*transition_chunk, initial_obs)
            loss_dict = self._model_step(chunk, initialize=index == 0)
            self._add_logs(totals, loss_dict)
        totals = self._mean_logs(totals, len(action_chunks))
        self.log_dict(
            {f"{self.model.__class__.__name__}/{key}/val": value
             for key, value in totals.items()},
            on_step=False,
            on_epoch=True,
            prog_bar="loss" in totals,
        )
        return totals["loss"]


class FullModelCheckpoint(pl.Callback):
    """Keep the existing rnn_epoch*.pth format for activity analysis."""

    def __init__(self, output_dir, every_n_epochs):
        self.output_dir = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.train_losses = []
        self.val_losses = []

    @staticmethod
    def _metric(metrics, suffix):
        for key, value in metrics.items():
            if key.endswith(suffix):
                return float(value.detach().cpu())
        return float("nan")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        self.train_losses.append(
            self._metric(trainer.callback_metrics, "/loss/train")
        )
        if self.every_n_epochs and epoch % self.every_n_epochs == 0:
            torch.save(
                pl_module.model,
                self.output_dir / f"rnn_epoch{epoch}.pth",
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.val_losses.append(
            self._metric(trainer.callback_metrics, "/loss/val")
        )

    def save_histories(self):
        np.save(self.output_dir / "loss_train.npy", np.asarray(self.train_losses))
        np.save(self.output_dir / "loss_test.npy", np.asarray(self.val_losses))


def train_with_lightning(args, model, train_loader, val_loader, output_dir):
    """Train an existing RSSM/MTRSSM WorldModel with pl.Trainer."""
    if args.hidden_reg != 0 or args.weights_reg != 0:
        raise ValueError(
            "Lightning world-model training currently requires "
            "hidden_reg=0 and weights_reg=0"
        )
    if args.architecture == "mtrssm" and args.high_kl_scale != 1.0:
        raise ValueError(
            "Lightning MTRSSM training currently requires high_kl_scale=1.0"
        )

    # Lightning's strategy teardown can move the wrapped module back to CPU.
    # Keep the device selected by RNNTrainer so the legacy evaluation and
    # activity code can safely feed its CUDA tensors to the returned model.
    model_device = next(model.parameters()).device

    checkpoint = FullModelCheckpoint(output_dir, args.save_model_every)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    logger = False
    if args.wandb:
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(experiment=__import__("wandb").run)

    runner = LightningWorldModelRunner(model, args.bptt_steps)
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision="32-true",
        logger=logger,
        callbacks=[checkpoint],
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )
    trainer.fit(
        model=runner,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    final_epoch = trainer.current_epoch
    torch.save(runner.model, Path(output_dir) / f"rnn_epoch{final_epoch}.pth")
    checkpoint.save_histories()
    return runner.model.to(model_device)
