import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

class Activation(nn.Module):
    def __init__(self, activation: str, reparametarize_fn: str = "gelu", devide_input: bool = False, **kwargs: Any):
        super().__init__()
        if "glu" not in activation.lower():
            kwargs.pop("dim", None)
        try:
            self.activation = getattr(nn, activation)(**kwargs)
        except AttributeError:
            if activation == "TanhExp":
                self.activation = TanhExp()
            elif activation == "REReLU":
                self.activation = REReLU(reparametarize_fn)
            elif activation == "TanhandREReLU":
                self.activation = TanhandREReLU(
                    reparametarize_fn, devide_input)
            else:
                raise NotImplementedError(
                    f"Activation: '{activation}' is not implemented yet."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class TanhandREReLU(nn.Module):
    def __init__(self, reparametarize_fn: str = "gelu", devide_input: bool = False) -> None:
        super().__init__()
        self.tanh = nn.Tanh()
        self.re_relu = REReLU(reparametarize_fn)
        self.devide_input = devide_input

    def forward(self, inputs: torch.Tensor):
        if self.devide_input:
            input_1, input_2 = inputs.chunk(2, dim=-1)
            tanh_result = self.tanh(input_1)
            re_relu_result = self.re_relu(input_2)
            return torch.cat([tanh_result, re_relu_result], dim=-1)
        else:
            tanh_result = self.tanh(inputs)
            re_relu_result = self.re_relu(inputs)
            return torch.cat([tanh_result, re_relu_result], dim=-1)


class TanhExp(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanhexp = TanhExpBase.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tanhexp(x)


class REReLU(nn.Module):
    def __init__(self, reparametarize_fn: str = "gelu") -> None:
        super().__init__()
        reparametarize_fn = reparametarize_fn.lower()
        self.reparametarize_fn = getattr(F, reparametarize_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.relu(x).detach()
            + self.reparametarize_fn(x)
            - self.reparametarize_fn(x).detach()
        )


class TanhExpBase(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor):

        return x * x.exp().tanh()

    @staticmethod
    def setup_context(
        ctx: Any, inputs: torch.Tensor, output: torch.Tensor
    ) -> Any:
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors

        grad_input = grad_output * (
            x.exp().tanh() - (x * x.exp() * (x.exp().tanh() ** 2 - 1))
        )
        return grad_input
