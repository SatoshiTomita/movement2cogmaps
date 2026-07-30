import os
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
from src.utils.utils import mytorch

def stack_dicts(dicts: Tuple[dict, ...], dim: int = 0) -> dict:
    return {key: mytorch.stack([d[key] for d in dicts], dim=dim) for key in dicts[0]}
def cat_dicts(dicts: Tuple[dict, ...], dim: int = 0) -> dict:
    return {key: mytorch.concat([d[key] for d in dicts], dim=dim) for key in dicts[0]}
def sum_dicts(dicts: Tuple[dict, ...]) -> dict:
    return {key: sum([d[key] for d in dicts]) for key in dicts[0]}

@dataclass
class ExpectedFreeEnergy:
    epistemic: torch.Tensor
    extrinsic: torch.Tensor
    efe: torch.Tensor

    def __len__(self):
        return self.epistemic.shape[0]

    def __getitem__(self, idx):
        return ExpectedFreeEnergy(
            self.epistemic[idx], self.extrinsic[idx], self.efe[idx]
        )

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/epistemic", self.epistemic.detach().cpu().numpy())
        np.save(f"{dir}/extrinsic", self.extrinsic.detach().cpu().numpy())
        np.save(f"{dir}/efe", self.efe.detach().cpu().numpy())

def stack_efe(efes: Tuple[ExpectedFreeEnergy, ...], dim: int = 0) -> ExpectedFreeEnergy:
    return ExpectedFreeEnergy(
        mytorch.stack([efe.epistemic for efe in efes], dim=dim),
        mytorch.stack([efe.extrinsic for efe in efes], dim=dim),
        mytorch.stack([efe.efe for efe in efes], dim=dim),
    )


@dataclass
class NormalStochNp:
    mean: np.ndarray
    std: np.ndarray
    stoch: np.ndarray

    def __getitem__(self, idx):
        return NormalStochNp(self.mean[idx], self.std[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/mean", self.mean)
        np.save(f"{dir}/std", self.std)
        np.save(f"{dir}/stoch", self.stoch)

    def __getattr__(self, name):
        return NormalStochNp(
            getattr(self.mean, name), getattr(self.std, name), getattr(self.stoch, name)
        )

    def reshape(self, shape: list):
        return NormalStochNp(
            self.mean.reshape(shape), self.std.reshape(shape), self.stoch.reshape(shape)
        )


@dataclass(frozen=True)
class NormalStoch:
    mean: torch.Tensor
    std: torch.Tensor
    stoch: torch.Tensor

    def __getitem__(self, idx):
        return NormalStoch(self.mean[idx], self.std[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        return self.stoch.shape

    def detach(self):
        return NormalStoch(self.mean.detach(), self.std.detach(), self.stoch.detach())

    def clone(self):
        return NormalStoch(self.mean.clone(), self.std.clone(), self.stoch.clone())

    def numpy(self):
        return NormalStochNp(
            self.mean.detach().cpu().numpy(),
            self.std.detach().cpu().numpy(),
            self.stoch.detach().cpu().numpy(),
        )

    def reshape(self, shape: list):
        return NormalStoch(
            self.mean.reshape(shape), self.std.reshape(shape), self.stoch.reshape(shape)
        )

    def flatten(self, start_dim: int, end_dim: int):
        return NormalStoch(
            self.mean.flatten(start_dim, end_dim),
            self.std.flatten(start_dim, end_dim),
            self.stoch.flatten(start_dim, end_dim),
        )

    def to(self, device):
        return NormalStoch(
            self.mean.to(device), self.std.to(device), self.stoch.to(device)
        )

    def permute(self, *dims):
        return NormalStoch(
            self.mean.permute(*dims), self.std.permute(*dims), self.stoch.permute(*dims)
        )

    def transpose(self, dim0: int, dim1: int):
        return NormalStoch(
            self.mean.transpose(dim0, dim1),
            self.std.transpose(dim0, dim1),
            self.stoch.transpose(dim0, dim1),
        )

    def unsqueeze(self, dim: int):
        return NormalStoch(
            self.mean.unsqueeze(dim), self.std.unsqueeze(dim), self.stoch.unsqueeze(dim)
        )

    def expand(self, expand_dim: int):
        return NormalStoch(
            self.mean.expand([expand_dim, *self.mean.shape[1:]]),
            self.std.expand([expand_dim, *self.std.shape[1:]]),
            self.stoch.expand([expand_dim, *self.stoch.shape[1:]]),
        )

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/mean", self.mean.detach().cpu().numpy())
        np.save(f"{dir}/std", self.std.detach().cpu().numpy())
        np.save(f"{dir}/stoch", self.stoch.detach().cpu().numpy())

    def __getattr__(self, name):
        return NormalStoch(
            getattr(self.mean, name), getattr(self.std, name), getattr(self.stoch, name)
        )


@dataclass
class CategoricStochNp:
    logits: np.ndarray
    probs: np.ndarray
    stoch: np.ndarray

    def __getitem__(self, idx):
        return CategoricStochNp(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/logits", self.logits)
        np.save(f"{dir}/probs", self.probs)
        np.save(f"{dir}/stoch", self.stoch)

    def __getattr__(self, name):
        return CategoricStochNp(
            getattr(self.logits, name),
            getattr(self.probs, name),
            getattr(self.stoch, name),
        )

    def reshape(self, shape: list):
        return CategoricStochNp(
            self.logits.reshape(shape),
            self.probs.reshape(shape),
            self.stoch.reshape(shape),
        )


@dataclass(frozen=True)
class CategoricStoch:
    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __getitem__(self, idx):
        return CategoricStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        return self.stoch.shape

    @property
    def mean(self):
        return self.probs

    def detach(self):
        return CategoricStoch(
            self.logits.detach(), self.probs.detach(), self.stoch.detach()
        )

    def clone(self):
        return CategoricStoch(
            self.logits.clone(), self.probs.clone(), self.stoch.clone()
        )

    def numpy(self):
        return CategoricStochNp(
            self.logits.detach().cpu().numpy(),
            self.probs.detach().cpu().numpy(),
            self.stoch.detach().cpu().numpy(),
        )

    def reshape(self, shape: list):
        return CategoricStoch(
            self.logits.reshape(shape),
            self.probs.reshape(shape),
            self.stoch.reshape(shape),
        )

    def flatten(self, start_dim: int, end_dim: int):
        return CategoricStoch(
            self.logits.flatten(start_dim, end_dim),
            self.probs.flatten(start_dim, end_dim),
            self.stoch.flatten(start_dim, end_dim),
        )

    def to(self, device):
        return CategoricStoch(
            self.logits.to(device), self.probs.to(device), self.stoch.to(device)
        )

    def permute(self, *dims):
        return CategoricStoch(
            self.logits.permute(*dims),
            self.probs.permute(*dims),
            self.stoch.permute(*dims),
        )

    def transpose(self, dim0: int, dim1: int):
        return CategoricStoch(
            self.logits.transpose(dim0, dim1),
            self.probs.transpose(dim0, dim1),
            self.stoch.transpose(dim0, dim1),
        )

    def unsqueeze(self, dim: int):
        return CategoricStoch(
            self.logits.unsqueeze(dim),
            self.probs.unsqueeze(dim),
            self.stoch.unsqueeze(dim),
        )

    def expand(self, expand_dim: int):
        return CategoricStoch(
            self.logits.expand([expand_dim, *self.logits.shape[1:]]),
            self.probs.expand([expand_dim, *self.probs.shape[1:]]),
            self.stoch.expand([expand_dim, *self.stoch.shape[1:]]),
        )

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/logits", self.logits.detach().cpu().numpy())
        np.save(f"{dir}/probs", self.probs.detach().cpu().numpy())
        np.save(f"{dir}/stoch", self.stoch.detach().cpu().numpy())

    def __getattr__(self, name):
        return CategoricStoch(
            getattr(self.logits, name),
            getattr(self.probs, name),
            getattr(self.stoch, name),
        )

@dataclass(frozen=True)
class BernoulliStoch:
    logits: torch.Tensor
    probs: torch.Tensor
    stoch: torch.Tensor

    def __getitem__(self, idx):
        return CategoricStoch(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    @property
    def shape(self):
        return self.stoch.shape

    @property
    def mean(self):
        return self.probs

    def detach(self):
        return BernoulliStoch(
            self.logits.detach(), self.probs.detach(), self.stoch.detach()
        )

    def clone(self):
        return BernoulliStoch(
            self.logits.clone(), self.probs.clone(), self.stoch.clone()
        )

    def numpy(self):
        return BernoulliStochNp(
            self.logits.detach().cpu().numpy(),
            self.probs.detach().cpu().numpy(),
            self.stoch.detach().cpu().numpy(),
        )

    def reshape(self, shape: list):
        return BernoulliStoch(
            self.logits.reshape(shape),
            self.probs.reshape(shape),
            self.stoch.reshape(shape),
        )

    def flatten(self, start_dim: int, end_dim: int):
        return BernoulliStoch(
            self.logits.flatten(start_dim, end_dim),
            self.probs.flatten(start_dim, end_dim),
            self.stoch.flatten(start_dim, end_dim),
        )

    def to(self, device):
        return BernoulliStoch(
            self.logits.to(device), self.probs.to(device), self.stoch.to(device)
        )

    def permute(self, *dims):
        return BernoulliStoch(
            self.logits.permute(*dims),
            self.probs.permute(*dims),
            self.stoch.permute(*dims),
        )

    def transpose(self, dim0: int, dim1: int):
        return BernoulliStoch(
            self.logits.transpose(dim0, dim1),
            self.probs.transpose(dim0, dim1),
            self.stoch.transpose(dim0, dim1),
        )

    def unsqueeze(self, dim: int):
        return BernoulliStoch(
            self.logits.unsqueeze(dim),
            self.probs.unsqueeze(dim),
            self.stoch.unsqueeze(dim),
        )

    def expand(self, expand_dim: int):
        return BernoulliStoch(
            self.logits.expand([expand_dim, *self.logits.shape[1:]]),
            self.probs.expand([expand_dim, *self.probs.shape[1:]]),
            self.stoch.expand([expand_dim, *self.stoch.shape[1:]]),
        )

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/logits", self.logits.detach().cpu().numpy())
        np.save(f"{dir}/probs", self.probs.detach().cpu().numpy())
        np.save(f"{dir}/stoch", self.stoch.detach().cpu().numpy())

    def __getattr__(self, name):
        return BernoulliStoch(
            getattr(self.logits, name),
            getattr(self.probs, name),
            getattr(self.stoch, name),
        )

@dataclass(frozen=True)
class BernoulliStochNp:
    logits: np.ndarray
    probs: np.ndarray
    stoch: np.ndarray

    def __getitem__(self, idx):
        return BernoulliStochNp(self.logits[idx], self.probs[idx], self.stoch[idx])

    def __len__(self):
        return self.stoch.shape[0]

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/logits", self.logits)
        np.save(f"{dir}/probs", self.probs)
        np.save(f"{dir}/stoch", self.stoch)

    def __getattr__(self, name):
        return BernoulliStochNp(
            getattr(self.logits, name), getattr(self.probs, name), getattr(self.stoch, name)
        )

    def reshape(self, shape: list):
        return BernoulliStochNp(
            self.logits.reshape(shape),
            self.probs.reshape(shape),
            self.stoch.reshape(shape),
        )

StochState = Union[NormalStoch, NormalStochNp, CategoricStoch, CategoricStochNp, BernoulliStoch, BernoulliStochNp]


class BernoulliStraightThrough(D.Bernoulli):
    has_rsample = True

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        samples = self.sample(sample_shape)
        probs = self.probs
        return samples + (probs - probs.detach())


@dataclass(frozen=True)
class Action:
    arm_dist: StochState
    gripper_dist: torch.Tensor
    entropy: torch.Tensor

    @property
    def arm_shape(self):
        return self.arm_dist.stoch.shape

    @property
    def gripper_shape(self):
        return self.gripper_dist.shape

    @property
    def action(self):
        if self.gripper_dist is not None:
            action = mytorch.concat(
                [self.arm_dist.stoch, torch.sigmoid(self.gripper_dist)], dim=-1
            )
        else:
            action = self.arm_dist.stoch

        return action

    @property
    def mean(self):
        if self.gripper_dist is not None:
            mean = mytorch.concat(
                [
                    (
                        self.arm_dist.mean.reshape(self.arm_shape)
                        if hasattr(self.arm_dist, "mean")
                        else self.arm_dist.probs.reshape(self.arm_shape)
                    ),
                    self.gripper_dist,
                ],
                dim=-1,
            )
        else:
            mean = (
                self.arm_dist.mean.reshape(self.arm_shape)
                if hasattr(self.arm_dist, "mean")
                else self.arm_dist.probs.reshape(self.arm_shape)
            )
        return mean

    def get_dist(self):
        return get_dist(self.arm_dist), (
            torch.sigmoid(self.gripper_dist) if self.gripper_dist is not None else None
        )


@dataclass(frozen=True)
class WorldStates:
    determ: Union[torch.Tensor, np.ndarray]
    prior: StochState
    posterior: StochState

    def __getitem__(self, idx):
        return WorldStates(
                self.determ[idx], 
                self.prior[idx], 
                self.posterior[idx]
                )

    def __len__(self):
        return self.determ.shape[0]

    @property
    def shape(self):
        return dict(
            determ=self.determ.shape,
            prior=self.prior.shape,
            posterior=self.posterior.shape,
        )
    @property
    def latent_states(self):
        return torch.cat([self.determ, self.posterior.stoch], dim=-1)

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/determ", self.determ.detach().cpu().numpy())
        self.prior.save(f"{dir}/prior")
        self.posterior.save(f"{dir}/posterior")

    def detach(self):
        return WorldStates(
            self.determ.detach(), self.prior.detach(), self.posterior.detach()
        )

    def clone(self):
        return WorldStates(
            self.determ.clone(), self.prior.clone(), self.posterior.clone()
        )

    def numpy(self):
        return WorldStates(
            self.determ.detach().cpu().numpy(),
            self.prior.numpy(),
            self.posterior.numpy(),
        )

    def flatten(self, start_dim: int = 0, end_dim: int = 1):
        return WorldStates(
            self.determ.flatten(start_dim, end_dim),
            self.prior.flatten(start_dim, end_dim),
            self.posterior.flatten(start_dim, end_dim),
        )

    def reshape_batch(self, batch_shape: list):
        return WorldStates(
            self.determ.reshape([*batch_shape, -1]),
            self.prior.reshape([*batch_shape, -1]),
            self.posterior.reshape([*batch_shape, -1]),
        )

    def to(self, device):
        return WorldStates(
            self.determ.to(device), self.prior.to(device), self.posterior.to(device)
        )

    def permute(self, *dims):
        return WorldStates(
            self.determ.permute(*dims),
            self.prior.permute(*dims),
            self.posterior.permute(*dims),
        )

    def transpose(self, dim0: int, dim1: int):
        return WorldStates(
            self.determ.transpose(dim0, dim1),
            self.prior.transpose(dim0, dim1),
            self.posterior.transpose(dim0, dim1),
        )
    
    def unsqueeze(self, dim: int):
        return WorldStates(
            self.determ.unsqueeze(dim),
            self.prior.unsqueeze(dim),
            self.posterior.unsqueeze(dim),
        )

    def expand(self, expand_dim: int):
        return WorldStates(
            self.determ.expand([expand_dim, *self.determ.shape[1:]]),
            self.prior.expand(expand_dim),
            self.posterior.expand(expand_dim),
        )


@dataclass(frozen=True)
class WorldStatesLayer:
    """Fast (layer0) + slow (layer1) world states for MTRSSM."""

    layer0: WorldStates
    layer1: WorldStates

    @property
    def latent_states(self):
        return torch.cat([self.layer0.latent_states, self.layer1.latent_states], dim=-1)

    @property
    def prior(self):
        return self.layer0.prior

    @property
    def posterior(self):
        return self.layer0.posterior

    @property
    def determ(self):
        return self.layer0.determ

    def detach(self):
        return WorldStatesLayer(self.layer0.detach(), self.layer1.detach())

    def to(self, device):
        return WorldStatesLayer(
            self.layer0.to(device),
            self.layer1.to(device),
        )

    def transpose(self, dim0: int, dim1: int):
        return WorldStatesLayer(
            self.layer0.transpose(dim0, dim1),
            self.layer1.transpose(dim0, dim1),
        )

    def numpy(self):
        return WorldStatesLayer(
            self.layer0.numpy(),
            self.layer1.numpy(),
        )


@dataclass(frozen=True)
class CoarseWorldStates:
    determ: torch.Tensor
    coarse: torch.Tensor
    prior: StochState
    posterior: StochState
    c_prior: StochState
    c_posterior: StochState
    gate: torch.Tensor

    def __getitem__(self, idx):
        return CoarseWorldStates(
            self.determ[idx],
            self.coarse[idx],
            self.prior[idx],
            self.posterior[idx],
            self.c_prior[idx],
            self.c_posterior[idx] if self.c_posterior is not None else None,
            self.gate[idx] if self.gate is not None else None,
        )
    def __len__(self):
        return self.determ.shape[0]

    def save(self, dir: str):
        os.makedirs(f"{dir}", exist_ok=True)
        np.save(f"{dir}/determ", self.determ.detach().cpu().numpy())
        np.save(f"{dir}/coarse", self.coarse.detach().cpu().numpy())
        self.prior.save(f"{dir}/prior")
        self.posterior.save(f"{dir}/posterior")
        self.c_prior.save(f"{dir}/c_prior")
        if self.c_posterior is not None:
            self.c_posterior.save(f"{dir}/c_posterior")
        if self.gate is not None:
            np.save(f"{dir}/gate", self.gate.detach().cpu().numpy())

    @property
    def latent_states(self):
        if self.c_posterior is None:
            return torch.cat([self.determ, self.posterior.stoch, self.coarse], dim=-1)
        else:
            return torch.cat([self.determ, self.posterior.stoch, self.coarse, self.c_posterior.stoch], dim=-1)

    @property
    def coarse_states(self):
        if self.c_posterior is None:
            return torch.cat([self.coarse, self.posterior.stoch], dim=-1)
        else:
            return torch.cat([self.coarse, self.c_posterior.stoch], dim=-1)

    def detach(self):
        return CoarseWorldStates(
            self.determ.detach(),
            self.coarse.detach(),
            self.prior.detach(),
            self.posterior.detach(),
            self.c_prior.detach(),
            self.c_posterior.detach() if self.c_posterior is not None else None,
            self.gate.detach() if self.gate is not None else None,
        )
    def clone(self):
        return CoarseWorldStates(
            self.determ.clone(),
            self.coarse.clone(),
            self.prior.clone(),
            self.posterior.clone(),
            self.c_prior.clone(),
            self.c_posterior.clone() if self.c_posterior is not None else None,
            self.gate.clone() if self.gate is not None else None,
        )
    def numpy(self):
        return CoarseWorldStates(
            self.determ.detach().cpu().numpy(),
            self.coarse.detach().cpu().numpy(),
            self.prior.numpy(),
            self.posterior.numpy(),
            self.c_prior.numpy(),
            self.c_posterior.numpy() if self.c_posterior is not None else None,
            self.gate.detach().cpu().numpy() if self.gate is not None else None,
        )
    def flatten(self, start_dim: int = 0, end_dim: int = 1):
        return CoarseWorldStates(
            self.determ.flatten(start_dim, end_dim),
            self.coarse.flatten(start_dim, end_dim),
            self.prior.flatten(start_dim, end_dim),
            self.posterior.flatten(start_dim, end_dim),
            self.c_prior.flatten(start_dim, end_dim),
            self.c_posterior.flatten(start_dim, end_dim) if self.c_posterior is not None else None,
            self.gate.flatten(start_dim, end_dim) if self.gate is not None else None,
        )
    def reshape_batch(self, batch_shape: list):
        return CoarseWorldStates(
            self.determ.reshape([*batch_shape, -1]),
            self.coarse.reshape([*batch_shape, -1]),
            self.prior.reshape([*batch_shape, -1]),
            self.posterior.reshape([*batch_shape, -1]),
            self.c_prior.reshape([*batch_shape, -1]),
            self.c_posterior.reshape([*batch_shape, -1]) if self.c_posterior is not None else None,
            self.gate.reshape([*batch_shape, -1]) if self.gate is not None else None,
        )
    def to(self, device):
        return CoarseWorldStates(
            self.determ.to(device),
            self.coarse.to(device),
            self.prior.to(device),
            self.posterior.to(device),
            self.c_prior.to(device),
            self.c_posterior.to(device) if self.c_posterior is not None else None,
            self.gate.to(device) if self.gate is not None else None,
        )
    def permute(self, *dims):
        return CoarseWorldStates(
            self.determ.permute(*dims),
            self.coarse.permute(*dims),
            self.prior.permute(*dims),
            self.posterior.permute(*dims),
            self.c_prior.permute(*dims),
            self.c_posterior.permute(*dims) if self.c_posterior is not None else None,
            self.gate.permute(*dims) if self.gate is not None else None,
        )
    def transpose(self, dim0: int, dim1: int):
        return CoarseWorldStates(
            self.determ.transpose(dim0, dim1),
            self.coarse.transpose(dim0, dim1),
            self.prior.transpose(dim0, dim1),
            self.posterior.transpose(dim0, dim1),
            self.c_prior.transpose(dim0, dim1),
            self.c_posterior.transpose(dim0, dim1) if self.c_posterior is not None else None,
            self.gate.transpose(dim0, dim1) if self.gate is not None else None,
        )

    def unsqueeze(self, dim: int):
        return CoarseWorldStates(
            self.determ.unsqueeze(dim),
            self.coarse.unsqueeze(dim),
            self.prior.unsqueeze(dim),
            self.posterior.unsqueeze(dim),
            self.c_prior.unsqueeze(dim),
            self.c_posterior.unsqueeze(dim) if self.c_posterior is not None else None,
            self.gate.unsqueeze(dim) if self.gate is not None else None,
        )
    
    def expand(self, expand_dim: int):
        return CoarseWorldStates(
            self.determ.expand([expand_dim, *self.determ.shape[1:]]),
            self.coarse.expand([expand_dim, *self.coarse.shape[1:]]),
            self.prior.expand(expand_dim),
            self.posterior.expand(expand_dim),
            self.c_prior.expand(expand_dim),
            self.c_posterior.expand(expand_dim) if self.c_posterior is not None else None,
            self.gate.expand([expand_dim, *self.gate.shape[1:]]) if self.gate is not None else None,
        )

def get_dist(state: StochState) -> D.Independent:
    if isinstance(state, NormalStoch):
        dist = D.Normal(state.mean, state.std)
        return D.Independent(dist, 1)
    elif isinstance(state, CategoricStoch):
        dist = D.OneHotCategoricalStraightThrough(probs=state.probs)
        return D.Independent(dist, 1)
    elif isinstance(state, BernoulliStoch):
        dist = BernoulliStraightThrough(probs=state.probs)
        return D.Independent(dist, 2)
    else:
        raise NotImplementedError


def cat_dist(stochs: Tuple[StochState, ...], dim: int = -1):
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            mytorch.concat([stoch.mean for stoch in stochs], dim=dim),
            mytorch.concat([stoch.std for stoch in stochs], dim=dim),
            mytorch.concat([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], CategoricStoch):
        return CategoricStoch(
            mytorch.concat([stoch.logits for stoch in stochs], dim=dim),
            mytorch.concat([stoch.probs for stoch in stochs], dim=dim),
            mytorch.concat([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            mytorch.concat([stoch.logits for stoch in stochs], dim=dim),
            mytorch.concat([stoch.probs for stoch in stochs], dim=dim),
            mytorch.concat([stoch.stoch for stoch in stochs], dim=dim),
        )
    else:
        return None


def stack_dist(stochs: Tuple[StochState, ...], dim: int = 0):
    if isinstance(stochs[0], NormalStoch):
        return NormalStoch(
            mytorch.stack([stoch.mean for stoch in stochs], dim=dim),
            mytorch.stack([stoch.std for stoch in stochs], dim=dim),
            mytorch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], CategoricStoch):
        return CategoricStoch(
            mytorch.stack([stoch.logits for stoch in stochs], dim=dim),
            mytorch.stack([stoch.probs for stoch in stochs], dim=dim),
            mytorch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    elif isinstance(stochs[0], BernoulliStoch):
        return BernoulliStoch(
            mytorch.stack([stoch.logits for stoch in stochs], dim=dim),
            mytorch.stack([stoch.probs for stoch in stochs], dim=dim),
            mytorch.stack([stoch.stoch for stoch in stochs], dim=dim),
        )
    else:
        return None


def cat_worlds(
    worlds: Union[Tuple[WorldStates, ...], Tuple[CoarseWorldStates, ...]], dim: int = -1
):
    if isinstance(worlds[0], WorldStates):
        return WorldStates(
                mytorch.concat([world.determ for world in worlds], dim=dim),
                cat_dist([world.prior for world in worlds], dim),
                cat_dist([world.posterior for world in worlds], dim),
            )

    elif isinstance(worlds[0], CoarseWorldStates):
        return CoarseWorldStates(
            mytorch.concat([world.determ for world in worlds], dim=dim),
            mytorch.concat([world.coarse for world in worlds], dim=dim),
            cat_dist([world.prior for world in worlds], dim),
            cat_dist([world.posterior for world in worlds], dim),
            cat_dist([world.c_prior for world in worlds], dim),
            cat_dist([world.c_posterior for world in worlds], dim),
            mytorch.concat([world.gate for world in worlds], dim=dim),
        )
    else:
        raise NotImplementedError


def stack_worlds(
    worlds: Union[
        Tuple[WorldStates, ...],
        Tuple[CoarseWorldStates, ...],
        Tuple[WorldStatesLayer, ...],
    ],
    dim: int = 0,
):
    if isinstance(worlds[0], WorldStatesLayer):
        return WorldStatesLayer(
            stack_worlds([world.layer0 for world in worlds], dim),
            stack_worlds([world.layer1 for world in worlds], dim),
        )
    if isinstance(worlds[0], WorldStates):
        return WorldStates(
                mytorch.stack([world.determ for world in worlds], dim=dim),
                stack_dist([world.prior for world in worlds], dim),
                stack_dist([world.posterior for world in worlds], dim),
            )
    elif isinstance(worlds[0], CoarseWorldStates):
        return CoarseWorldStates(
            mytorch.stack([world.determ for world in worlds], dim=dim),
            mytorch.stack([world.coarse for world in worlds], dim=dim),
            stack_dist([world.prior for world in worlds], dim),
            stack_dist([world.posterior for world in worlds], dim),
            stack_dist([world.c_prior for world in worlds], dim),
            stack_dist([world.c_posterior for world in worlds], dim),
            mytorch.stack([world.gate for world in worlds], dim=dim),
        )
    else:
        raise NotImplementedError

Worlds = Union[WorldStates, CoarseWorldStates, WorldStatesLayer]

