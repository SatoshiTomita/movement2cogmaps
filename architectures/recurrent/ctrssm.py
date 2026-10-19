"""Continuous-time recurrent state-space model used by ``RNN_experiment``.

The stochastic-state machinery is the same as the single-level RSSM, while
the deterministic state is updated by the Euler-discretised CTRNN cell used by
the standalone CTRNN baseline.  A carried BPTT state contains the membrane
potential, exposed activity, and posterior stochastic state.  Only the latter
two are returned as latent activity for downstream spatial analyses.
"""

from dataclasses import asdict, is_dataclass

import torch
from torch import nn

from architectures.recurrent.ctrnn_bptt import CTRNNCell
from networks.distributions import Representation, Transition
from utils.config import RSSMConfig
from utils.states import WorldStates, stack_worlds


class CTRSSMPredictor(nn.Module):
    """RSSM whose deterministic transition is a continuous-time RNN.

    Time convention matches :class:`RSSMPredictor`: ``output[:, t]``
    reconstructs the pre-action observation, then ``action[:, t]`` and the
    post-action observation update the state for the next time point.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        cfg: RSSMConfig,
        tau: float = 4.0,
        nonlinearity: str = "sigmoid",
        dropouts=(0.0, 0.0, 0.0),
        bias: bool = False,
        gridcells_dim: int | None = None,
    ):
        super().__init__()
        if len(dropouts) != 3:
            raise ValueError(
                "CTRSSM dropouts must contain [input, recurrent, decoder]"
            )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.determ_dim = cfg.determ_dim
        self.encoder = nn.Identity()

        stoch_kwargs = (
            asdict(cfg.stoch_cfg)
            if is_dataclass(cfg.stoch_cfg)
            else cfg.stoch_cfg
        )
        self.prior = Transition(self.determ_dim, **stoch_kwargs)
        self.posterior = Representation(
            obs_dim, self.determ_dim, **stoch_kwargs
        )
        if self.prior.stoch_dim != self.posterior.stoch_dim:
            raise ValueError("CTRSSM prior and posterior dimensions differ")

        self.stoch_dim = self.prior.stoch_dim
        self.latent_dim = self.determ_dim + self.stoch_dim
        self.state_dim = 2 * self.determ_dim + self.stoch_dim
        self.use_stoch = "posterior"
        self.latent_dim_for_action = self.latent_dim

        self.ctrnn_cell = CTRNNCell(
            input_dim=action_dim + self.stoch_dim,
            hidden_dim=self.determ_dim,
            tau=tau,
            nonlinearity=nonlinearity,
            dropouts=dropouts[:2],
            bias=bias,
        )

        # Match the ordinary RSSM's observation-conditioned initial state,
        # while treating the learned value as the CTRNN membrane potential.
        initializer_dim = max(obs_dim, self.determ_dim)
        self.initializer = nn.Sequential(
            nn.Linear(obs_dim, initializer_dim),
            nn.Mish(),
            nn.Linear(initializer_dim, self.determ_dim),
        )
        self.gc2membrane = (
            nn.Linear(gridcells_dim, self.determ_dim)
            if gridcells_dim is not None else None
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, obs_dim, bias=False),
            nn.Dropout(dropouts[2]),
        )

        self.membrane = None
        self.hidden_state = None
        self.prev_stoch = None

    def forward(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        initial_obs: torch.Tensor | None = None,
        gridcells: torch.Tensor | None = None,
    ):
        return self.observe(action, observation, state, initial_obs, gridcells)

    def _packed_state(self) -> torch.Tensor:
        return torch.cat(
            [self.membrane, self.hidden_state, self.prev_stoch], dim=-1
        )

    def _latent_state(self) -> torch.Tensor:
        return torch.cat([self.hidden_state, self.prev_stoch], dim=-1)

    def _initialize_state(
        self,
        batch_size: int,
        initial_obs: torch.Tensor,
        gridcells: torch.Tensor | None = None,
    ) -> None:
        if initial_obs.shape != (batch_size, self.obs_dim):
            raise ValueError(
                "initial_obs shape must be "
                f"{(batch_size, self.obs_dim)}, got {tuple(initial_obs.shape)}"
            )
        embedded_obs = self.encoder(initial_obs)
        if self.gc2membrane is None:
            self.membrane = self.initializer(embedded_obs)
        else:
            if gridcells is None:
                raise ValueError(
                    "gridcells must be provided when grid-cell input is enabled"
                )
            self.membrane = self.gc2membrane(gridcells[:, 0])
        self.hidden_state = self.ctrnn_cell.activation_fn(self.membrane)
        self.prev_stoch = self.posterior(
            self.hidden_state, embedded_obs
        ).stoch

    def _restore_state(self, state: torch.Tensor, batch_size: int) -> None:
        expected_shape = (batch_size, self.state_dim)
        if state.shape != expected_shape:
            raise ValueError(
                f"state shape must be {expected_shape}, got {tuple(state.shape)}"
            )
        self.membrane, self.hidden_state, self.prev_stoch = state.split(
            [self.determ_dim, self.determ_dim, self.stoch_dim], dim=-1
        )

    def _step(
        self, action: torch.Tensor, observation: torch.Tensor | None
    ) -> WorldStates:
        transition_input = torch.cat([action, self.prev_stoch], dim=-1)
        self.hidden_state, ctrnn_state = self.ctrnn_cell(
            transition_input,
            (self.membrane, self.hidden_state),
        )
        self.membrane, self.hidden_state = ctrnn_state

        prior = self.prior(self.hidden_state)
        posterior = (
            self.posterior(self.hidden_state, observation)
            if observation is not None
            else prior
        )
        self.prev_stoch = (
            posterior.stoch if observation is not None else prior.stoch
        )
        return WorldStates(self.hidden_state, prior, posterior)

    def observe(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        initial_obs: torch.Tensor | None = None,
        gridcells: torch.Tensor | None = None,
    ):
        """Process a BPTT window and return reconstructions and states.

        Args:
            action: ``[B,T,A]`` actions from time t to t+1.
            observation: ``[B,T,O]`` post-action observations.
            state: optional packed ``[membrane, activity, stochastic]`` state.
            initial_obs: pre-action observation used when ``state`` is absent.
        """
        if action.ndim != 3 or observation.ndim != 3:
            raise ValueError("action and observation must have shape [B,T,D]")
        if action.shape[:2] != observation.shape[:2]:
            raise ValueError(
                "action and observation batch/time dimensions must match"
            )
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action dimension must be {self.action_dim}, "
                f"got {action.shape[-1]}"
            )
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observation dimension must be {self.obs_dim}, "
                f"got {observation.shape[-1]}"
            )

        batch_size, window_size = action.shape[:2]
        if window_size == 0:
            raise ValueError("CTRSSM cannot observe an empty time window")

        if state is None:
            if initial_obs is None:
                raise ValueError("initial_obs must be provided if state is None")
            self._initialize_state(batch_size, initial_obs, gridcells)
        else:
            self._restore_state(state, batch_size)

        current_latent = self._latent_state()
        world_history = []
        next_latents = []
        action_tbd = action.transpose(0, 1)
        observation_tbd = self.encoder(observation).transpose(0, 1)

        for step in range(window_size):
            world = self._step(action_tbd[step], observation_tbd[step])
            world_history.append(world)
            next_latents.append(
                torch.cat([world.determ, world.posterior.stoch], dim=-1)
            )

        worlds = stack_worlds(world_history)
        next_latent_tbd = torch.stack(next_latents, dim=0)
        reconstruction_latent_tbd = torch.cat(
            [current_latent.unsqueeze(0), next_latent_tbd[:-1]], dim=0
        )

        outputs = self.decoder(reconstruction_latent_tbd).transpose(0, 1)
        hidden_all = reconstruction_latent_tbd.transpose(0, 1)
        hidden_last = self._packed_state()

        self.last_prior = worlds.prior
        self.last_posterior = worlds.posterior
        self.last_aux_loss = None

        return outputs, hidden_all, hidden_last
