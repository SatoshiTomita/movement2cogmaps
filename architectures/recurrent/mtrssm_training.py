"""Training adapter for the unchanged two-level :class:`MTRSSM` model."""

from dataclasses import replace

import torch
from torch import nn

from architectures.recurrent.rssm import MTRSSM, RSSM
from utils.config import MTRSSMConfig
from utils.states import stack_dist


class MTRSSMPredictor(MTRSSM):
    """Add observation decoding and BPTT state handling to ``MTRSSM``.

    The inherited ``MTRSSM`` implementation is intentionally left unchanged.
    This adapter builds the same low/high RSSM hierarchy, while exposing the
    ``observe`` interface required by the recurrent training loop.

    Time convention for index ``t``:
        * decode the current packed state into ``output[t] = o_hat_t``;
        * update the low level with ``action[t] = a_t`` and
          ``observation[t] = o_(t+1)``;
        * periodically update the high level from the new low-level state.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: MTRSSMConfig):
        # MTRSSM.__init__ targets an older RSSM constructor. Build its declared
        # hierarchy here without changing the MTRSSM class itself.
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.temporal_abstraction = cfg.temporal_abstraction
        if self.temporal_abstraction < 1:
            raise ValueError("temporal_abstraction must be at least 1")

        self.top_obs = cfg.top_obs
        lower_cfg = replace(cfg.lower_cfg, rnn_name="MTRNN")
        higher_cfg = replace(cfg.higher_cfg, rnn_name="MTRNN")

        higher_stoch_dim = (
            higher_cfg.stoch_cfg.stoch_dim * higher_cfg.stoch_cfg.n_class
        )
        self.low_level = RSSM(
            obs_dim=obs_dim,
            input_dim=action_dim + higher_stoch_dim,
            cfg=lower_cfg,
        )

        if self.top_obs == "determ":
            top_obs_dim = self.low_level.determ_dim
        elif self.top_obs == "stoch":
            top_obs_dim = self.low_level.stoch_dim
        elif self.top_obs == "both":
            top_obs_dim = self.low_level.latent_dim
        else:
            raise ValueError(
                "top_obs must be 'determ', 'stoch', or 'both', "
                f"but got {self.top_obs!r}"
            )

        self.high_level = RSSM(
            obs_dim=top_obs_dim,
            input_dim=0,
            cfg=higher_cfg,
        )

        self.stoch_dim = self.low_level.stoch_dim
        self.determ_dim = self.low_level.determ_dim
        self.latent_dim = self.determ_dim + self.stoch_dim
        self.higher_stoch_dim = self.high_level.stoch_dim
        self.higher_determ_dim = self.high_level.determ_dim
        self.higher_latent_dim = (
            self.higher_determ_dim + self.higher_stoch_dim
        )
        self.latent_dim_for_action = self.latent_dim + self.higher_latent_dim

        self.encoder = nn.Identity()
        # 再構成の入力は下位層の決定的状態と確率的状態の両方で入力
        self.decoder = nn.Linear(
            self.latent_dim,
            obs_dim,
            bias=False,
        )
        self._temporal_phase = 0

    def forward(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        initial_obs: torch.Tensor | None = None,
    ):
        return self.observe(action, observation, state, initial_obs)

    def _top_observation(self, low_state):
        if self.top_obs == "determ":
            return low_state.determ
        if self.top_obs == "stoch":
            return low_state.posterior.stoch
        return torch.cat(
            [low_state.determ, low_state.posterior.stoch],
            dim=-1,
        )

    def _packed_state(self):
        return torch.cat(
            [
                self.low_level.hidden_state,
                self.low_level.prev_stoch,
                self.high_level.hidden_state,
                self.high_level.prev_stoch,
            ],
            dim=-1,
        )

    def _decoder_state(self, packed_state: torch.Tensor) -> torch.Tensor:
        """Select the low-level deterministic and stochastic decoder state."""
        if packed_state.shape[-1] != self.latent_dim_for_action:
            raise ValueError(
                "packed state has an unexpected final dimension: "
                f"expected {self.latent_dim_for_action}, "
                f"got {packed_state.shape[-1]}"
            )
        return packed_state[..., :self.latent_dim]

    @staticmethod
    def _restore_mtrnn_internal_state(layer, determ_state):
        """Restore the MTRNN membrane state from its exposed tanh state."""
        if not hasattr(layer.rnn, "hidden"):
            return
        if getattr(layer.rnn, "apply_tanh", False):
            eps = torch.finfo(determ_state.dtype).eps
            layer.rnn.hidden = torch.atanh(
                determ_state.clamp(min=-1 + eps, max=1 - eps)
            )
        else:
            layer.rnn.hidden = determ_state

    def _restore_state(self, state: torch.Tensor):
        expected = self.latent_dim_for_action
        if state.ndim != 2 or state.shape[-1] != expected:
            raise ValueError(
                f"state shape must be [B,{expected}], but got {tuple(state.shape)}"
            )

        split_sizes = [
            self.determ_dim,
            self.stoch_dim,
            self.higher_determ_dim,
            self.higher_stoch_dim,
        ]
        low_h, low_z, high_h, high_z = state.split(split_sizes, dim=-1)
        self.low_level.hidden_state = low_h
        self.low_level.prev_stoch = low_z
        self.high_level.hidden_state = high_h
        self.high_level.prev_stoch = high_z
        self._restore_mtrnn_internal_state(self.low_level, low_h)
        self._restore_mtrnn_internal_state(self.high_level, high_h)

    def _initialize_state(self, batch_size: int, initial_obs: torch.Tensor):
        initial_embed = self.encoder(initial_obs)
        self.low_level.init_latent(batch_size, initial_embed)

        if self.top_obs == "determ":
            top_obs = self.low_level.hidden_state
        elif self.top_obs == "stoch":
            top_obs = self.low_level.prev_stoch
        else:
            top_obs = torch.cat(
                [self.low_level.hidden_state, self.low_level.prev_stoch],
                dim=-1,
            )
        self.high_level.init_latent(batch_size, top_obs)
        self._temporal_phase = 0

    def observe(
        self,
        action: torch.Tensor,
        observation: torch.Tensor,
        state: torch.Tensor | None = None,
        initial_obs: torch.Tensor | None = None,
    ):
        """Observe a BPTT window and reconstruct its pre-action images.

        Args:
            action: ``[B,T,A]`` actions from time t to t+1.
            observation: ``[B,T,O]`` post-action images at time t+1.
            state: optional packed low/high state at the window's first time.
            initial_obs: current image at the first time when state is absent.

        Returns:
            outputs: ``[B,T,O]`` reconstructions at time t.
            hidden_all: ``[B,T,L]`` packed low/high states at time t.
            hidden_last: ``[B,L]`` state after all T actions.
        """
        if action.ndim != 3 or observation.ndim != 3:
            raise ValueError("action and observation must have shape [B,T,D]")
        if action.shape[:2] != observation.shape[:2]:
            raise ValueError(
                "action and observation batch/time dimensions must match"
            )

        batch_size, window_size = action.shape[:2]
        if window_size == 0:
            raise ValueError("MTRSSM cannot observe an empty time window")

        if state is None:
            if initial_obs is None:
                raise ValueError("initial_obs must be provided if state is None")
            self._initialize_state(batch_size, initial_obs)
        else:
            if state.shape[0] != batch_size:
                raise ValueError(
                    "state and action batch dimensions must match"
                )
            self._restore_state(state)

        current_latent = self._packed_state()
        next_latents = []
        low_priors, low_posteriors = [], []
        high_priors, high_posteriors = [], []

        action_tbd = action.transpose(0, 1)
        observation_tbd = self.encoder(observation).transpose(0, 1)

        for offset in range(window_size):
            action_t = action_tbd[offset]
            next_obs_t = observation_tbd[offset]
            # Combine the action with the slow stochastic context.
            low_input = torch.cat(
                [action_t, self.high_level.prev_stoch],
                dim=-1,
            )
            # RSSM.step also supplies the previous low stochastic state and
            # recurrent deterministic state to the fast MTRNN.
            low_state, _ = self.low_level.step(low_input, next_obs_t)
            low_priors.append(low_state.prior)
            low_posteriors.append(low_state.posterior)

            if self._temporal_phase == 0:
                top_obs = self._top_observation(low_state)
                no_action = action_t.new_empty((batch_size, 0))
                high_state, _ = self.high_level.step(no_action, top_obs)
                high_priors.append(high_state.prior)
                high_posteriors.append(high_state.posterior)

            next_latents.append(self._packed_state())
            self._temporal_phase = (
                self._temporal_phase + 1
            ) % self.temporal_abstraction

        next_latent_tbd = torch.stack(next_latents, dim=0)
        reconstruction_latent_tbd = torch.cat(
            [current_latent.unsqueeze(0), next_latent_tbd[:-1]],
            dim=0,
        )

        low_reconstruction_latent_tbd = self._decoder_state(
            reconstruction_latent_tbd
        )
        outputs = self.decoder(low_reconstruction_latent_tbd).transpose(0, 1)
        hidden_all = reconstruction_latent_tbd.transpose(0, 1)
        hidden_last = next_latent_tbd[-1]

        self.last_prior = stack_dist(low_priors)
        self.last_posterior = stack_dist(low_posteriors)
        self.last_high_prior = (
            stack_dist(high_priors) if high_priors else None
        )
        self.last_high_posterior = (
            stack_dist(high_posteriors) if high_posteriors else None
        )

        return outputs, hidden_all, hidden_last
