import torch


def get_hidden_l2norm(hidden_all: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm of hidden states across the batch and time dimensions."""
    return torch.norm(hidden_all, p=2, dim=-1).mean()


def get_weights_l2norm(model: torch.nn.Module) -> torch.Tensor:
    """Sum of L2 norms over all weight parameters in the model."""
    norms = [torch.norm(param, p=2) for name, param in model.named_parameters() if 'weight' in name]
    if not norms:
        raise ValueError("No weights found in the model.")
    return sum(norms)


class DiscountLoss(torch.nn.Module):
    """Applies an exponentially decaying discount to multi-step prediction losses.

    Args:
        loss_fn: Element-wise loss function (reduction='none').
        discount_factor: Discount rate applied per future step.
        n_future_pred: Number of future prediction steps.
    """

    def __init__(self, loss_fn, discount_factor, n_future_pred):
        super().__init__()
        self.loss_fn = loss_fn

        self.pow = torch.nn.parameter.Parameter(
            torch.pow(discount_factor, torch.arange(n_future_pred)),
            requires_grad=False
        )

    def forward(self, outputs, labels):
        """Compute discounted loss.

        For multi-step predictions (>3 dims), averages over batch/time/features
        then applies the discount vector. For single-step, returns plain mean.
        """
        loss = self.loss_fn(outputs, labels)
        # average over batch samples, time steps and features
        if len(loss.shape) > 3:
            loss = torch.mean(loss, dim=(0, 2, 3))
            return torch.sum(loss * self.pow) / self.pow.sum()
        return torch.mean(loss)
