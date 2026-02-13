import torch
import torch.nn.functional as F
import math


class MexicanHat(torch.nn.Module):
    """Difference-of-Gaussians (DoG) activation resembling a Mexican-hat profile.

    For x <= delta the output follows the DoG curve (normalised so peak = 1);
    for x > delta the output is clamped to 1.

    Args:
        normalize: Optional post-activation normalisation ('softmax' or 'sum').
    """

    def __init__(self, normalize=None):
        super().__init__()
        self.pi = torch.tensor(math.pi)
        self.width = torch.tensor(0.9)
        self.surround_scale = torch.tensor(2.0)
        self.ws2 = (self.width * self.surround_scale) ** 2
        # delta: zero-crossing so that y(0)=0 and y(delta)=1
        self.delta = torch.sqrt(
            -4 * torch.log(self.surround_scale) * self.ws2
            / (1 - self.surround_scale ** 2)
        )

        if normalize is not None and normalize not in ('softmax', 'sum'):
            raise ValueError("normalize must be 'softmax' or 'sum'")
        self.normalize = normalize

        self.softmax = torch.nn.Softmax(dim=1)

    def _normalize(self, y):
        if self.normalize == 'softmax':
            return self.softmax(y)
        if self.normalize == 'sum':
            y = y + torch.abs(y.min(dim=1).values)[..., None]
            return y / y.sum(dim=1)[..., None]
        return y

    def _compute_dog(self, x):
        """Compute the Difference-of-Gaussians profile."""
        E = torch.exp(-(x - self.delta) ** 2 / (2 * self.width ** 2))
        I = torch.exp(-(x - self.delta) ** 2 / (2 * self.ws2))

        y = (
            E / (2*self.pi* self.width**2) -
            I / (2*self.pi*self.ws2)
        ) * ( # normalize to have y=1 when x=1
            2*self.pi*self.ws2 /
            (self.surround_scale**2 -1)
        )

        if y.isnan().any():
            raise ValueError("MexicanHat activation function produced NaN values.")
        return torch.where(x <= self.delta, y, torch.ones_like(y))

    def forward(self, x):
        y = self._compute_dog(x)
        return self._normalize(y)


class MexicanHatStandard(torch.nn.Module):
    """Standard (Ricker wavelet) Mexican-hat activation: (1 - x^2) * exp(-x^2/2).

    Args:
        normalize: Optional post-activation normalisation ('softmax' or 'sum').
    """

    def __init__(self, normalize=None):
        super().__init__()
        if normalize is not None and normalize not in ('softmax', 'sum'):
            raise ValueError("normalize must be 'softmax' or 'sum'")
        self.normalize = normalize
        self.softmax = torch.nn.Softmax(dim=1)

    def _normalize(self, y):
        if self.normalize == 'softmax':
            return self.softmax(y)
        if self.normalize == 'sum':
            y = y + torch.abs(y.min(dim=1).values)[..., None]
            return y / y.sum(dim=1)[..., None]
        return y

    def forward(self, x):
        y = (1 - x**2)*torch.exp(-x**2 / 2)
        return self._normalize(y)


class HardSoftmax(torch.nn.Module):
    """Softmax followed by a hard threshold that zeros out small activations.

    Values below ``1 / (0.5 * latent_dim)`` are set to zero.

    Args:
        latent_dim: Dimensionality used to compute the threshold.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.threshold = 1 / (0.5 * latent_dim)

    def forward(self, x):
        s = torch.nn.Softmax(dim=1)(x)
        return s * (s >= self.threshold).float()


class HardSigmoid(torch.nn.Module):
    """Piece-wise linear approximation of sigmoid: clamp(0.2x + 0.5, 0, 1)."""

    def forward(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)


class _STEFunction(torch.autograd.Function):
    """Heaviside step with straight-through-estimator gradient."""

    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(torch.nn.Module):
    """Binary step activation using a straight-through estimator for gradients."""

    def forward(self, x):
        return _STEFunction.apply(x)