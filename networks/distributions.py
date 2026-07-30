from typing import Dict, Literal, Tuple, Union
from einops import rearrange
import numpy as np

import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from networks.activations import Activation
from utils.states import (CategoricStoch, NormalStoch, BernoulliStoch, BernoulliStraightThrough)
from utils.utils import mytorch
# from src.networks.vq import ProbabristicLFQ, AuxLoss, BSQCodebook
from networks.layers import States2Map2d, States2Map1d, MLPLayer
from utils.config import MLPConfig, States2Map1dConfig, States2Map2dConfig


class Representation(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        determ_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        dist: Literal["normal", "tanh_normal", "categorical", "binary"],
        layers: int = 1,
        activation: str = "Mish",
        n_class: int = 1,
        temprature: float = 1.0,
        spherical: bool = False,
        deterministic_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        assert layers >= 1, "layers must be greater than 1"

        self.determ_dim = determ_dim
        self.stoch_dim = stoch_dim * n_class
        self._stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.dist = dist
        self.layers = layers
        self.act_fn = Activation(activation)
        self.n_class = n_class
        self.temprature = torch.tensor(temprature).float()
        self.spherical = spherical
        self.deterministic_ratio = deterministic_ratio
        print("dist", dist)

        self.dense = self._build_dense()
        if dist == "normal":
            self.posterior = self.normal
        elif dist == "tanh_normal":
            self.posterior = self.tanh_normal
        elif dist == "categorical":
            self.posterior = self.categorical
        elif dist == "binary":
            self.posterior = self.binary
        else:
            raise NotImplementedError

        # if spherical:
        #     self.codebook = BSQCodebook(n_class)

        self._softplus = nn.Softplus()

    def _build_dense(self):
        dense = []
        dense += [
            nn.Linear(
                self.determ_dim + self.obs_dim, self.hidden_dim
            )
        ]
        dense += [self.act_fn]
        for i in range(self.layers-1):
            dense += [
                nn.Linear(
                    self.hidden_dim, self.hidden_dim
                )
            ]
            dense += [self.act_fn]
        if "normal" in self.dist:
            dense += [nn.Linear(self.hidden_dim, self.stoch_dim * 2)]
        else:
            dense += [nn.Linear(self.hidden_dim, self.stoch_dim)]
        return nn.Sequential(*dense)

    def softplus(self, x):
        return self._softplus(x) + 0.1

    def normal(self, deterministic_state, observation, deterministic=False, inv_tmp: float = 1.0):

        stochastic = self.dense(
            torch.cat([deterministic_state, observation], dim=-1))
        mu, std = torch.chunk(stochastic, 2, dim=-1)
        std = self.softplus(std)

        posterior_dist = D.Normal(mu, std)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample() if not deterministic else mu

        posterior = NormalStoch(mu, std, sample if not deterministic else mu)

        return posterior

    def tanh_normal(self, deterministic_state, observation, deterministic=False, inv_tmp: float = 1.0):
        stochastic = self.dense(
            torch.cat([deterministic_state, observation], dim=-1))
        mu, std = torch.chunk(stochastic, 2, dim=-1)
        mu = torch.tanh(mu)
        std = self.softplus(std)
        # print(mu)

        posterior_dist = D.Normal(mu, std)
        posterior_dist = D.Independent(posterior_dist, 1)

        sample = posterior_dist.rsample() if not deterministic else mu

        posterior = NormalStoch(mu, std, sample if not deterministic else mu)

        return posterior

    def categorical(self, deterministic_state, observation, deterministic=False, inv_tmp: float = 1.0):
        # print(deterministic_state.shape)
        # print(observation.shape)
        logits = self.dense(
            torch.cat([deterministic_state, observation], dim=-1))
        batch_shape = logits.shape[:-1]
        logits = torch.chunk(logits, self._stoch_dim, dim=-1)
        logits = torch.stack(logits, dim=-2)
        logits = logits * inv_tmp
        probs = mytorch.softmax(logits, dim=-1, temperature=self.temprature)
        if self.deterministic_ratio:
            determ_dim = int(self.deterministic_ratio * self._stoch_dim)
            determ_prob, stoch_prob = probs.split([determ_dim, self._stoch_dim - determ_dim], dim=-2)
            determ_sample_idx = determ_prob.argmax(dim=-1)
            determ_sample = F.one_hot(determ_sample_idx, num_classes=determ_prob.shape[-1])
            determ_sample = determ_sample + determ_prob - determ_prob.detach()

            posterior_dist = D.OneHotCategoricalStraightThrough(probs=stoch_prob)
            posterior_dist = D.Independent(posterior_dist, 1)
            stoch_sample = posterior_dist.rsample()
            sample = torch.cat([determ_sample, stoch_sample], dim=-2)


        else:
            posterior_dist = D.OneHotCategoricalStraightThrough(probs=probs)
            posterior_dist = D.Independent(posterior_dist, 1)

            sample = posterior_dist.rsample()

        if self.spherical:
            sample = sample * 2 - 1

        posterior = CategoricStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) if not deterministic else self.deterministic_onehot(probs).reshape([*batch_shape, -1])
            )

        return posterior

    def binary(self, deterministic_state, observation, deterministic=False, inv_tmp: float = 1.0):
        logits = self.dense(
            torch.cat([deterministic_state, observation], dim=-1))
        batch_shape = logits.shape[:-1]
        logits = torch.chunk(logits, self._stoch_dim, dim=-1)
        logits = torch.stack(logits, dim=-2)
        logits = logits * inv_tmp
        probs = torch.sigmoid(logits)

        if self.deterministic_ratio:
            determ_dim = int(self.deterministic_ratio * self._stoch_dim)
            determ_prob, stoch_prob = probs.split([determ_dim, self._stoch_dim - determ_dim], dim=-2)
            determ_sample = torch.where(determ_prob > 0.5, 1.0, 0.0)
            determ_sample = determ_sample + determ_prob - determ_prob.detach()

            posterior_dist = BernoulliStraightThrough(probs=stoch_prob)
            posterior_dist = D.Independent(posterior_dist, 1)
            stoch_sample = posterior_dist.rsample()
            sample = torch.cat([determ_sample, stoch_sample], dim=-2)
        else:
            posterior_dist = BernoulliStraightThrough(probs=probs)  
            posterior_dist = D.Independent(posterior_dist, 1)

            sample = posterior_dist.rsample()

        # if self.spherical:
        #     sample = self.codebook.bits_to_codes(sample)

        posterior = BernoulliStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) if not deterministic else self.deterministic_onehot(probs).reshape([*batch_shape, -1])
        )

        return posterior



    def forward(self, deterministic_state, observation, deterministic=False, inv_tmp: float = 1.0):
        return self.posterior(deterministic_state, observation, deterministic=deterministic, inv_tmp=inv_tmp)

    def deterministic_onehot(self, input:torch.Tensor):
        return F.one_hot(input.argmax(dim=-1), num_classes=self.classes) + input - input.detach()


class Transition(nn.Module):
    def __init__(
        self,
        determ_dim: int,
        stoch_dim: int,
        hidden_dim: int,
        dist: Literal["normal", "tanh_normal", "categorical", "binary"],
        layers,
        activation,
        n_class: int = 1,
        temprature: float = 1.0,
        spherical: bool = False,
        deterministic_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        assert layers >= 1, "layers must be greater than 1"

        self.determ_dim = determ_dim
        self.stoch_dim = stoch_dim * n_class
        self._stoch_dim = stoch_dim
        self.hidden_dim = hidden_dim
        self.dist = dist
        self.layers = layers
        self.act_fn = Activation(activation)
        self.n_class = n_class
        self.temprature = torch.tensor(temprature).float()
        self.spherical = spherical
        self.deterministic_ratio = deterministic_ratio

        self.dense = self._build_dense()
        if dist == "normal":
            self.prior = self.normal
        elif dist == "tanh_normal":
            self.prior = self.tanh_normal
        elif dist == "categorical":
            self.prior = self.categorical
        elif dist == "binary":
            self.prior = self.binary
        else:
            raise NotImplementedError
        
        # if spherical:
        #     self.codebook = BSQCodebook(n_class)

        self._softplus = nn.Softplus()

    def softplus(self, x):
        return self._softplus(x) + 0.1

    def _build_dense(self):
        dense = []
        dense += [nn.Linear(self.determ_dim, self.hidden_dim)]
        dense += [self.act_fn]
        for i in range(self.layers-1):
            dense += [nn.Linear(self.hidden_dim, self.hidden_dim)]
            dense += [self.act_fn]
        if "normal" in self.dist:
            dense += [nn.Linear(self.hidden_dim, self.stoch_dim * 2)]
        else:
            dense += [nn.Linear(self.hidden_dim, self.stoch_dim)]
        return nn.Sequential(*dense)

    def normal(self, deterministic_state, deterministic=False):
        stochastic = self.dense(deterministic_state)
        mu, std = torch.chunk(stochastic, 2, dim=-1)
        std = self.softplus(std)

        prior_dist = D.Normal(mu, std)
        prior_dist = D.Independent(prior_dist, 1)

        prior = NormalStoch(mu, std, prior_dist.rsample() if not deterministic else mu)

        return prior

    def tanh_normal(self, deterministic_state, deterministic=False):
        stochastic = self.dense(deterministic_state)
        mu, std = torch.chunk(stochastic, 2, dim=-1)
        mu = torch.tanh(mu)
        std = self.softplus(std)
        # print(mu)

        prior_dist = D.Normal(mu, std)
        prior_dist = D.Independent(prior_dist, 1)

        prior = NormalStoch(mu, std, prior_dist.rsample() if not deterministic else mu)

        return prior

    def categorical(self, deterministic_state, deterministic=False):
        logits = self.dense(deterministic_state)
        batch_shape = logits.shape[:-1]
        logits = torch.chunk(logits, self._stoch_dim, dim=-1)
        logits = torch.stack(logits, dim=-2)
        probs = mytorch.softmax(logits, dim=-1, temperature=self.temprature)
        if self.deterministic_ratio:
            determ_dim = int(self.deterministic_ratio * self._stoch_dim)
            determ_prob, stoch_prob = probs.split([determ_dim, self._stoch_dim - determ_dim], dim=-2)
            determ_sample_idx = determ_prob.argmax(dim=-1)
            determ_sample = F.one_hot(determ_sample_idx, num_classes=determ_prob.shape[-1])
            determ_sample = determ_sample + determ_prob - determ_prob.detach()

            prior_dist = D.OneHotCategoricalStraightThrough(probs=stoch_prob)
            prior_dist = D.Independent(prior_dist, 1)
            stoch_sample = prior_dist.rsample()
            sample = torch.cat([determ_sample, stoch_sample], dim=-2)
        else:
            prior_dist = D.OneHotCategoricalStraightThrough(probs=probs)
            
            prior_dist = D.Independent(prior_dist, 1)

            sample = prior_dist.rsample()

        if self.spherical:
            sample = sample * 2 - 1

        prior = CategoricStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) if not deterministic else self.deterministic_onehot(probs).reshape([*batch_shape, -1])
        )

        return prior

    def binary(self, deterministic_state, deterministic=False):
        logits = self.dense(deterministic_state)
        batch_shape = logits.shape[:-1]
        logits = torch.chunk(logits, self._stoch_dim, dim=-1)
        logits = torch.stack(logits, dim=-2)
        probs = torch.sigmoid(logits)

        if self.deterministic_ratio:
            determ_dim = int(self.deterministic_ratio * self._stoch_dim)
            determ_prob, stoch_prob = probs.split([determ_dim, self._stoch_dim - determ_dim], dim=-2)
            determ_sample = torch.where(determ_prob > 0.5, 1.0, 0.0)
            determ_sample = determ_sample + determ_prob - determ_prob.detach()

            prior_dist = BernoulliStraightThrough(probs=stoch_prob)
            prior_dist = D.Independent(prior_dist, 1)
            stoch_sample = prior_dist.rsample()
            sample = torch.cat([determ_sample, stoch_sample], dim=-2)
        else:
            prior_dist = BernoulliStraightThrough(probs=probs)  
            prior_dist = D.Independent(prior_dist, 1)

            sample = prior_dist.rsample()

        # if self.spherical:
        #     sample = self.codebook.bits_to_codes(sample)

        prior = BernoulliStoch(
            logits, probs, sample.reshape([*batch_shape, -1]) if not deterministic else self.deterministic_onehot(probs).reshape([*batch_shape, -1])
        )

        return prior

    def forward(self, deterministic_state, deterministic=False):
        return self.prior(deterministic_state, deterministic=deterministic)

    def deterministic_onehot(self, input:torch.Tensor):
        return F.one_hot(input.argmax(dim=-1), num_classes=self.classes) + input - input.detach()
