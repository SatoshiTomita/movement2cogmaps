from abc import ABC, abstractmethod
import numpy as np


class Trainer(ABC):
    """Abstract base class for epoch-based trainers.

    Provides helper methods for accumulating losses, gradient/weight norms,
    and hidden-layer statistics into a logging dictionary.

    Args:
        optimizer: Torch optimizer.
        loss_fn: Loss function.
        device: Torch device.
    """

    def __init__(self, optimizer, loss_fn, device):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def _update_dict(self, prefix, names, values, d):
        """Accumulate *values* into dict *d* under ``prefix/name`` keys."""
        for name, v in zip(names, values):
            key = f"{prefix}/{name}"
            val = v if isinstance(v, (int, float)) else v.detach().item()
            d[key] = d.get(key, 0) + val
        return d

    def _update_losses(self, names, losses, d):
        """Accumulate loss scalars into *d* under the ``loss/`` prefix."""
        return self._update_dict("loss", names, losses, d)

    def _update_hidden_layer(self, names, values, d):
        """Accumulate hidden-layer statistics under the ``hidden_layer/`` prefix."""
        return self._update_dict("hidden_layer", names, values, d)

    def _update_norms(self, model, d):
        """Accumulate gradient and weight L2 norms for all model parameters."""
        for name, p in model.named_parameters():
            if p.grad is not None:
                key = f"gradients/{name}"
                d[key] = d.get(key, 0) + p.grad.detach().norm(2).item()
            if "weight" in name:
                key = f"weights/{name}"
                d[key] = d.get(key, 0) + p.detach().norm(2).item()
        return d

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def test_epoch(self):
        pass


class EarlyStopping:
    """Two-phase early stopping: first reduces learning rate, then stops training.

    Monitors a validation loss history. When the loss is predominantly increasing
    or plateauing over a tolerance window, the ``reduced_lr`` flag is set. If the
    condition triggers again after LR reduction, ``early_stop`` is set.

    Args:
        lr_tolerance: Number of recent epochs to check before reducing LR.
        es_tolerance: Number of recent epochs to check before stopping.
        min_delta: Absolute tolerance for detecting a plateau.
    """

    def __init__(self, lr_tolerance, es_tolerance, min_delta):
        self.lr_tolerance = lr_tolerance
        self.es_tolerance = es_tolerance
        self.min_delta = min_delta

        self.reduced_lr = False
        self.early_stop = False

    def __call__(self, val_loss_list):
        """Check *val_loss_list* and update ``reduced_lr`` / ``early_stop`` flags."""
        tolerance = self.es_tolerance if self.reduced_lr else self.lr_tolerance
        if len(val_loss_list) < max(self.lr_tolerance, self.es_tolerance):
            return

        recent = np.array(val_loss_list[-tolerance:])

        # Check for predominantly increasing loss
        n_increases = np.sum(recent[1:] > recent[:-1])
        if n_increases >= int(0.55 * tolerance):
            if self.reduced_lr:
                self.early_stop = True
            self.reduced_lr = True
            return

        # Check for plateauing loss
        if np.allclose(recent[:-1], recent[1:], rtol=0, atol=self.min_delta):
            if self.reduced_lr:
                self.early_stop = True
            self.reduced_lr = True
