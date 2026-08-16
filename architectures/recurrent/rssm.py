import torch
from utils.states import (CategoricStoch, NormalStoch, WorldStates, WorldStatesLayer,
                              CoarseWorldStates, Worlds, stack_worlds, stack_dicts)
from utils.config import MTRSSMConfig, RSSMConfig
from dataclasses import asdict, is_dataclass
from utils.utils import mytorch
import torch.nn as nn
from networks.distributions import Representation, Transition
import networks.rnn as rnn

class RSSMPredictor(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg: RSSMConfig):
        """"
         input:

        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 最初は観測を変換せず、そのままRSSMに入力する
        self.encoder=nn.Identity()

        # 潜在状態を更新するRSSM
        self.rssm = RSSM(obs_dim=obs_dim, input_dim=action_dim, cfg=cfg)

        # [h,z]から観測を再構成
        self.decoder=nn.Linear(
            in_features=self.rssm.latent_dim,
            out_features=obs_dim,
            bias=False
        )

    def forward(self,action:torch.Tensor,observation:torch.Tensor,state:torch.Tensor |None=None,initial_obs:torch.Tensor|None=None):
        return self.observe(
            action,
            observation=observation,
            state=state,
            initial_obs=initial_obs,
        )

    def observe(self,action:torch.Tensor,observation:torch.Tensor,state:torch.Tensor|None=None,initial_obs:torch.Tensor|None=None):
        """
        Args:
            action[B,T,action_dim]: Transitions from time t to t+1.
            observation[B,T,obs_dim]: Post-action observations at time t+1.
            state[B,latent_dim]: State at the first, pre-action time t.
            initial_obs[B,obs_dim]: Observation used to infer the first state
                when no carried state is available.
        
        Returns:
            outputs[B,T,obs_dim]: Reconstructions at the pre-action times.
            hidden_all[B,T,latent_dim]: States matching outputs in time.
            hidden_last[B,latent_dim]: Final post-action state for carry-over.
        """

        batch_size = action.shape[0]
        # The decoder reconstructs the observation at the current state before
        # applying action[t].  Keep that state separately from the T states
        # produced by the T transitions below.
        if state is None:
            if initial_obs is None:
                raise ValueError("initial_obs must be provided if state is None")

            initial_embed=self.encoder(initial_obs)

            current_latent = self.rssm.init_latent(
                batch_size=batch_size,
                obs=initial_embed,
            )
        else:
            if state.shape !=(
                batch_size,
                self.rssm.latent_dim
            ):
                raise ValueError(f"state shape must be {(batch_size,self.rssm.latent_dim)}, but got {state.shape}")

            self.rssm.hidden_state=(
                state[:,:self.rssm.determ_dim]
            )

            self.rssm.prev_stoch=(
                state[:,self.rssm.determ_dim:]
            )
            current_latent = state

        # 観測を埋め込みへ変換
        embed_obs = self.encoder(observation)

        # actionとembed_obsの次元を入れ替える ([B,T,D]->[T,B,D])
        action_tbd=action.transpose(0,1) 
        embed_obs_tbd=embed_obs.transpose(0,1)

        # rssmに通して潜在状態を計算する
        # 
        worlds,aux_loss=self.rssm(
            action=action_tbd,
            embed_obs=embed_obs_tbd,
        )

        # Each transition state corresponds to the observation after action[t]:
        # [state_(t+1), ..., state_(t+T)].
        next_latent_tbd = torch.cat(
            [
                worlds.determ,
                worlds.posterior.stoch,
            ],
            dim=-1,
        )

        # Reconstruct the pre-action observations from
        # [state_t, ..., state_(t+T-1)].  The final transition state is retained
        # only as the recurrent state carried into the next BPTT window.
        reconstruction_latent_tbd = torch.cat(
            [
                current_latent.unsqueeze(0),
                next_latent_tbd[:-1],
            ],
            dim=0,
        )

        # decoderに通す[T,B,H+Z]→[T,B,obs_dim]
        outputs_tbd=self.decoder(reconstruction_latent_tbd)

        # [B,T,D]へ再び戻す
        outputs=outputs_tbd.transpose(0,1)
        hidden_all=reconstruction_latent_tbd.transpose(0,1)
        hidden_last=next_latent_tbd[-1]

        # observeで計算したprior,posterior,補助損失をRSSMPredictorの属性として保存
        self.last_prior=worlds.prior
        self.last_posterior=worlds.posterior
        self.last_aux_loss=aux_loss

        return outputs,hidden_all,hidden_last

            


class RSSM(nn.Module):
    """
      input:shape(B,T,scene_dim+action_dim)
      output:
        - world_history: shape(T, B, H+Z)
        - loss_history
    """
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


class MTRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        input_dim: int,
        cfg: MTRSSMConfig,
        **kwargs,
    ):
        super().__init__()
        self.temporal_abstraction = cfg.temporal_abstraction

        self.low_level = RSSM(
            obs_dim,
            input_dim+(cfg.higher_cfg.stoch_cfg.stoch_dim *
                       cfg.higher_cfg.stoch_cfg.n_class),
            rnn_name="MTRNN",
            **asdict(cfg.lower_cfg) if is_dataclass(cfg.lower_cfg) else cfg.lower_cfg
        )
        self.top_obs = cfg.top_obs
        if cfg.top_obs == "both":

            top_obs_dim = self.low_level.latent_dim
        elif cfg.top_obs == "determ":
            top_obs_dim = self.low_level.determ_dim
        elif cfg.top_obs == "stoch":
            top_obs_dim = self.low_level.stoch_dim

        self.high_level = RSSM(
            top_obs_dim,
            0,
            rnn_name="MTRNN",
            **asdict(cfg.higher_cfg) if is_dataclass(cfg.higher_cfg) else cfg.higher_cfg
        )
        self.use_stoch = "posterior"

        self.stoch_dim = self.low_level.stoch_dim
        self.determ_dim = self.low_level.determ_dim
        self.latent_dim = self.determ_dim + self.stoch_dim

        self.higher_stoch_dim = self.high_level.stoch_dim
        self.higher_determ_dim = self.high_level.determ_dim
        self.higher_latent_dim = self.higher_determ_dim + self.higher_stoch_dim
        self.latent_dim_for_action = self.latent_dim + self.higher_latent_dim

    def init_latent(self, batch_size, obs=None):

        # for i in range(self.layers):
        # exec(f'obs = self.layer_{i}.init_latent(batch_size, obs)')
        init_latent0 = self.low_level.init_latent(batch_size, obs)

        if self.top_obs == "determ":
            obs = self.low_level.hidden_state
        elif self.top_obs == "stoch":
            obs = obs
        elif self.top_obs == "both":
            obs = mytorch.concat([self.low_level.hidden_state, obs], dim=-1)
        else:
            raise NotImplementedError
        init_latent1 = self.high_level.init_latent(batch_size, obs)
        return torch.cat([init_latent0, init_latent1], dim=-1)

    def set_prev_states(self, worlds: Worlds):

        self.low_level.hidden_state = prev_determs[0]
        self.low_level.prev_stoch = prev_determs[0]
        self.high_level.hidden_state = prev_determs[1]
        self.high_level.prev_stoch = prev_stochs[1]

        return torch.cat([prev_determs[0], prev_determs[0], prev_determs[1], prev_determs[1]], dim=-1)

    def step(self, action, obs=None, timestep: int = 0):
        layers = []

        inputs = mytorch.concat([action, self.high_level.prev_stoch], dim=-1)

        world_states = self.low_level.step(inputs, obs)

        layers.append(world_states)
        obs = world_states.layer0.posterior.stoch if obs is not None else world_states.layer0.prior.stoch

        if self.top_obs == "determ":
            obs = world_states.determ
        elif self.top_obs == "stoch":
            obs = obs
        elif self.top_obs == "both":
            obs = mytorch.concat([world_states.layer0.determ, obs], dim=-1)
        else:
            raise NotImplementedError

        if timestep % self.temporal_abstraction == 0:

            world_states = self.high_level.step(None, obs)
            layers.append(world_states)

        # print(all_world_states.layer_1. is None)
        return WorldStatesLayer(layers[0], layers[1])

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor):
        """
        Args:
            action: shape(T, B, D)
            embed_obs: shape(T, B, D)

        """

        world_history = []

        for t in range(len(action)):
            world_states = self.step(action[t], embed_obs[t], t)

            world_history.append(world_states)

        world_history = stack_worlds(world_history)

        return world_history
