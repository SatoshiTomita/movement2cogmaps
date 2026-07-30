import torch
from utils.states import (CategoricStoch, NormalStoch, WorldStates, WorldStatesLayer,
                              CoarseWorldStates, Worlds, stack_worlds, stack_dicts)
from utils.config import RSSMConfig
from dataclasses import asdict, is_dataclass
from utils.utils import mytorch
import torch.nn as nn
from networks.distributions import Representation, Transition
import networks.rnn as rnn

class RSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        input_dim: int,
        cfg: RSSMConfig
    ):
        super().__init__()
        self.coarse_obs = None

        self.rnn = getattr(rnn, f"{cfg.rnn_name}Cell")(
            input_dim + (cfg.stoch_cfg.stoch_dim*cfg.stoch_cfg.n_class),
            cfg.determ_dim,
            obs_dim if cfg.init_from_ == "obs" else cfg.init_from_,
            **asdict(cfg.rnn_cfg) if is_dataclass(cfg.rnn_cfg) else cfg.rnn_cfg,
        )
        self._is_vqrnn = "VQ" in cfg.rnn_name
        self._init_from_ = cfg.init_from_
        self._init_with_ = cfg.init_with_

        if cfg.stoch_cfg.stoch_dim:
            self.prior = Transition(
                cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg)
            self.posterior = Representation(
                obs_dim, cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg
            )
            assert self.prior.stoch_dim == self.posterior.stoch_dim

        self.stoch_dim = self.prior.stoch_dim if cfg.stoch_cfg.stoch_dim else 0
        self.determ_dim = cfg.determ_dim
        self.latent_dim = cfg.determ_dim + self.stoch_dim
        self.use_stoch = "posterior"
        self.latent_dim_for_action = self.latent_dim

    def init_latent(self, batch_size, obs=None):
        self.hidden_state = self.rnn.init_latent(
            obs if self._init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)

        if self._init_with_ == "posterior":
            self.prev_stoch = self.posterior.forward(
                self.hidden_state, obs).stoch.reshape([batch_size, -1])
        elif self.stoch_dim > 0:
            self.prev_stoch = self.prior(self.hidden_state).stoch.reshape([batch_size, -1])
        else:
            self.prev_stoch = None

        return mytorch.concat([self.hidden_state, self.prev_stoch], dim=-1)

    def set_prev_states(self, worlds: Worlds):
        self.hidden_state = worlds.determ
        self.prev_stoch = worlds.posterior.stoch
        return torch.cat([self.hidden_state, self.prev_stoch], dim=-1)

    def detach(self):
        """Detach recurrent states between truncated BPTT chunks."""
        if hasattr(self, "hidden_state") and self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if hasattr(self, "prev_stoch") and self.prev_stoch is not None:
            self.prev_stoch = self.prev_stoch.detach()
        if hasattr(self.rnn, "detach"):
            self.rnn.detach()

    def step(self, action, obs=None, timestep: int = 0) -> WorldStates:
        determ_state = self.rnn(mytorch.concat(
            [action, self.prev_stoch], dim=-1), self.hidden_state)

        if self._is_vqrnn:
            self.hidden_state, vq_loss = determ_state
            loss_dict = {"vq_loss": vq_loss}
        else:
            self.hidden_state = determ_state
            loss_dict = None

        if self.stoch_dim:
            prior = self.prior(determ_state)
            posterior = self.posterior(
                determ_state, obs) if obs is not None else prior
        else:
            prior = NormalStoch()
            posterior = NormalStoch()
        states = WorldStates(determ_state, prior, posterior)
        self.prev_stoch = states.posterior.stoch if obs is not None else states.prior.stoch

        return states, loss_dict

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor):
        """
        Args:
            action: shape(T, B, D)
            embed_obs: shape(T, B, D)

        """

        world_history = []
        loss_history = []

        for t in range(len(action)):
            world_states, loss = self.step(action[t], embed_obs[t], t)
            world_history.append(world_states)
            loss_history.append(loss)
        world_history = stack_worlds(world_history)
        if self._is_vqrnn:
            loss_history = stack_dicts(loss_history)
        else:
            loss_history = None

        return world_history, loss_history