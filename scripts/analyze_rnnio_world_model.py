#!/usr/bin/env python3
"""Run analyze_world_model.py on the 0824base_rnnio checkpoints.

The rnnio variant feeds the flattened 16x32 frame directly to the CRSSMV4
dynamics (no CNN encoder) and reconstructs it with one bias-free linear layer
from the 240-dimensional latent state. Its training code and config were not
kept, so the architecture is reconstructed from the checkpoint parameter
shapes: the CNN encoder and ResNet decoder built from
models/cfg/0824base_rnnio.yaml are swapped for these modules before loading.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze_world_model  # noqa: E402


class FlattenEncoder(nn.Module):
    """Identity observation encoder: (..., C, H, W) -> (..., C*H*W)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=-3)


class LinearDecoder(nn.Module):
    """Bias-free linear read-out of the latent state, reshaped to a frame."""

    def __init__(self, latent_dim: int, obs_shape: tuple[int, ...]):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        n_pixels = int(torch.tensor(self.obs_shape).prod())
        self.linear = nn.Linear(latent_dim, n_pixels, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.linear(z)
        return out.reshape(*z.shape[:-1], *self.obs_shape)


_build_world = analyze_world_model.build_world


def build_rnnio_world(config_node, obs_shape, action_dim):
    world = _build_world(config_node, obs_shape, action_dim)
    n_pixels = int(torch.tensor(obs_shape).prod())
    if world.embed_obs_dim != n_pixels:
        raise ValueError(
            f"embed_obs_dim must equal the pixel count {n_pixels}; "
            f"got {world.embed_obs_dim}"
        )
    world.obs_encoder = FlattenEncoder(n_pixels)
    world.obs_decoder = LinearDecoder(world.dynamics.latent_dim, obs_shape)
    return world


if __name__ == "__main__":
    analyze_world_model.build_world = build_rnnio_world
    raise SystemExit(analyze_world_model.main(analyze_world_model.parse_args()))
