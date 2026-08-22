
from dataclasses import asdict, is_dataclass
from typing import Dict, Tuple

import networks.rnn as rnn
import torch
import torch.nn as nn
from einops import rearrange
from utils.config import (
    DistributionConfig, GateL0RDConfig, 
    MTRNNConfig, CRSSMConfig, CRSSMV4Config,
    MTRSSMConfig, RSSMConfig, 
    Map2StatesConfig, MLPConfig)
from utils.states import (CategoricStoch, NormalStoch, WorldStates, WorldStatesLayer,
                              CoarseWorldStates, Worlds,
                              stack_worlds, stack_dicts)
from utils.utils import mytorch
from networks.layers import Map2States, MLPLayer

from networks.distributions import Representation, Transition
import numpy as np



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
        self.hidden_state = self.hidden_state.detach()
        self.prev_stoch = self.prev_stoch.detach()
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

        # RSSMの現行コンストラクタはRSSMConfigを受け取るため、設定を展開せずそのまま渡す。
        self.low_level = RSSM(
            obs_dim,
            input_dim+(cfg.higher_cfg.stoch_cfg.stoch_dim *
                       cfg.higher_cfg.stoch_cfg.n_class),
            cfg.lower_cfg,
        )
        self.top_obs = cfg.top_obs
        if cfg.top_obs == "both":

            top_obs_dim = self.low_level.latent_dim
        elif cfg.top_obs == "determ":
            top_obs_dim = self.low_level.determ_dim
        elif cfg.top_obs == "stoch":
            top_obs_dim = self.low_level.stoch_dim

        # 上位層も下位層と同じRSSM APIで構築し、入力次元だけを上位観測に合わせる。
        self.high_level = RSSM(
            top_obs_dim,
            0,
            cfg.higher_cfg,
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

        # 上位層の初期状態は、初期化済みの下位h/zを観測として生成する。
        if self.top_obs == "determ":
            obs = self.low_level.hidden_state
        elif self.top_obs == "stoch":
            obs = self.low_level.prev_stoch
        elif self.top_obs == "both":
            obs = mytorch.concat([self.low_level.hidden_state, self.low_level.prev_stoch], dim=-1)
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

    def detach(self):
        # WorldModelのBPTT窓の境界で、上下両方の再帰グラフを切り離す。
        self.low_level.detach()
        self.high_level.detach()

    def step(self, action, obs=None, timestep: int = 0):
        layers = []

        # 下位層は行動に前時刻の上位確率状態を加え、上位文脈を使って更新する。
        inputs = mytorch.concat([action, self.high_level.prev_stoch], dim=-1)
        world_states, _ = self.low_level.step(inputs, obs)
        layers.append(world_states)

        obs = (
            world_states.posterior.stoch
            if obs is not None
            else world_states.prior.stoch
        )
        if self.top_obs == "determ":
            obs = world_states.determ
        elif self.top_obs == "stoch":
            obs = obs
        elif self.top_obs == "both":
            obs = mytorch.concat([world_states.determ, obs], dim=-1)
        else:
            raise NotImplementedError

        # 元の実装意図を保ち、上位層は指定された時間間隔でのみ更新する。
        if timestep % self.temporal_abstraction == 0:
            empty_action = action.new_empty((action.shape[0], 0))
            world_states, _ = self.high_level.step(
                empty_action, obs
            )
            self._last_high_state = world_states

        layers.append(self._last_high_state)

        # WorldModelが要求する上下層の状態と補助損失の組を返す。
        return WorldStatesLayer(layers[0], layers[1]), None

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor):
        """
        Args:
            action: shape(T, B, D)
            embed_obs: shape(T, B, D)

        """

        world_history = []

        for t in range(len(action)):
            world_states, _ = self.step(action[t], embed_obs[t], t)
            world_history.append(world_states)

        world_history = stack_worlds(world_history)

        # WorldModelのdynamics.forward共通形式に合わせる。
        return world_history, None

class CRSSM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        input_dim: int,
        cfg: CRSSMConfig,
    ):
        super().__init__()

        self.cfg = cfg

        self.init_from_ = cfg.init_from_
        self.init_with_ = cfg.init_with_
        self.coarse_dyn = getattr(rnn, f"{cfg.coarse_rnn}Cell")(
                (cfg.stoch_cfg.stoch_dim*cfg.stoch_cfg.n_class),
                cfg.coarse_dim,
                obs_dim if cfg.init_from_ == "obs" else cfg.init_from_,
                **asdict(cfg.coarse_cfg) if is_dataclass(cfg.coarse_cfg) else cfg.coarse_cfg,
        )
        self._c_is_vqrnn = "VQ" in cfg.coarse_rnn
        self.coarse_dim = cfg.coarse_dim * getattr(cfg.coarse_cfg, "n_class", 1)
        self.c_prior = Transition(
            self.coarse_dim, 
            **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg)
        
        self.coarse_stoch_dim = self.c_prior.stoch_dim if cfg.stoch_cfg.stoch_dim else 0

        self.precise_dyn = getattr(rnn, f"{cfg.precise_rnn}Cell")(
            input_dim + self.coarse_dim + (cfg.stoch_cfg.stoch_dim*cfg.stoch_cfg.n_class),
            cfg.determ_dim,
            obs_dim if cfg.init_from_ == "obs" else cfg.init_from_,
            **asdict(cfg.precise_cfg) if is_dataclass(cfg.precise_cfg) else cfg.precise_cfg,
        )
        self._d_is_vqrnn = "VQ" in cfg.precise_rnn
        self.d_prior = Transition(
            cfg.determ_dim+self.coarse_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg)
        self.d_posterior = Representation(
            obs_dim, cfg.determ_dim+self.coarse_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg
        )


        assert (
            self.d_prior.stoch_dim
            == self.d_posterior.stoch_dim
        ), "stoch_dim must be the same for all stochastic layers"

        self.stoch_dim = self.d_prior.stoch_dim if cfg.stoch_cfg.stoch_dim else 0
        self.determ_dim = cfg.determ_dim
        self.latent_dim = cfg.determ_dim + self.stoch_dim + self.coarse_dim 
        self.coarse_latent_dim = self.coarse_dim + self.stoch_dim
        self.use_stoch = "posterior"
        self.latent_dim_for_action = self.latent_dim
        self.coarse_obs = "obs"


    def init_latent(self, batch_size, obs=None):
        self.coarse_state = self.coarse_dyn.init_latent(
            obs if self.init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)

        self.hidden_state = self.precise_dyn.init_latent(
            obs if self.init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)
        if self.init_with_ == "posterior":
            d_posterior = self.posterior(
                    self.hidden_state, 
                    obs) 
            self.prev_stoch = d_posterior.stoch
        elif self.stoch_dim:
            self.prev_stoch = self.d_prior(torch.cat([self.hidden_state, self.coarse_state], dim=-1)).stoch
        else:
            raise NotImplementedError("Only init with posterior or prior is supported")
        return torch.cat([self.hidden_state, self.prev_stoch, self.coarse_state], dim=-1)

    def set_prev_states(self, worlds: Worlds, set_mtrnn_hidden=True):
        self.hidden_state = worlds.determ
        self.prev_stoch = worlds.posterior.stoch
        self.coarse_state = worlds.coarse
        if isinstance(self.precise_dyn, rnn.MTRNNCell) and set_mtrnn_hidden:
            self.precise_dyn.hidden = torch.cat(self.precise_dyn.hidden_histopy).unsqueeze(0).expand(
                    self.hidden_state.shape[0]//len(self.precise_dyn.hidden_histopy), -1, -1).flatten(0, 1)
        if isinstance(self.coarse_dyn, rnn.MTRNNCell) and set_mtrnn_hidden:
            self.coarse_dyn.hidden = torch.cat(self.coarse_dyn.hidden_histopy).unsqueeze(0).expand(
                    self.coarse_state.shape[0]//len(self.coarse_dyn.hidden_histopy), -1, -1).flatten(0, 1)

        return torch.cat([self.hidden_state, self.prev_stoch, self.coarse_state], dim=-1)
    
    def step(self, action: torch.Tensor, obs: torch.Tensor =None, deterministic:bool = False) -> Tuple[CoarseWorldStates, Dict[str, torch.Tensor]]:
        loss_dict = {}
        coarse_returns = self.coarse_dyn(self.prev_stoch, self.coarse_state)
        if self._c_is_vqrnn:
            self.coarse_state, c_vq_loss, gate = coarse_returns
            loss_dict["c_vq_loss"] = c_vq_loss
        else:
            self.coarse_state, gate = coarse_returns
        c_prior = self.c_prior(
            self.coarse_state, deterministic=deterministic)

        determ_state = self.precise_dyn(
            torch.cat([action, self.prev_stoch, self.coarse_state], dim=-1), self.hidden_state
        )
        if self._d_is_vqrnn:
            self.hidden_state, d_vq_loss = determ_state
            loss_dict["d_vq_loss"] = d_vq_loss
        else:
            self.hidden_state = determ_state

        d_prior = self.d_prior(
            torch.cat([self.hidden_state, self.coarse_state], dim=-1), 
            deterministic=deterministic)
        d_posterior = self.d_posterior(
            torch.cat([self.hidden_state, self.coarse_state], dim=-1),
            obs, 
            deterministic=deterministic) if obs is not None else d_prior
        states = CoarseWorldStates(self.hidden_state, self.coarse_state,
                             d_prior, d_posterior, c_prior, None, gate)

        self.prev_stoch = states.posterior.stoch if obs is not None else states.prior.stoch

        return states, loss_dict

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor, deterministic:bool = False) -> Tuple[Worlds, Dict[str, torch.Tensor]]:
        """
        Args:
            action: shape(T, B, D)
            embed_obs: shape(T, B, D)

        """

        world_history = []
        loss_history = []

        for t in range(len(action)):
            world_states, loss_dict = self.step(action[t], embed_obs[t], deterministic)
            world_history.append(world_states)
            loss_history.append(loss_dict)
        world_history = stack_worlds(world_history)
        if self._c_is_vqrnn or self._d_is_vqrnn:
            loss_history = stack_dicts(loss_history)
        else:
            loss_history = None

        return world_history, loss_history

class CRSSMV4(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        input_dim: int,
        cfg: CRSSMV4Config,
    ):
        super().__init__()

        self.cfg = cfg

        self.init_from_ = cfg.init_from_
        self.init_with_ = cfg.init_with_
        self.coarse_dyn = getattr(rnn, f"{cfg.coarse_rnn}Cell")(
                (cfg.coarse_stoch_cfg.stoch_dim*cfg.coarse_stoch_cfg.n_class),
                cfg.coarse_dim,
                obs_dim if cfg.init_from_ == "obs" else cfg.init_from_,
                **asdict(cfg.coarse_cfg) if is_dataclass(cfg.coarse_cfg) else cfg.coarse_cfg,
        )
        self._c_is_vqrnn = "VQ" in cfg.coarse_rnn
        self.coarse_dim = cfg.coarse_dim * getattr(cfg.coarse_cfg, "n_class", 1)
        self.c_prior = Transition(
            self.coarse_dim, 
            **asdict(cfg.coarse_stoch_cfg) if is_dataclass(cfg.coarse_stoch_cfg) else cfg.coarse_stoch_cfg)
        self.c_posterior = Representation(
            obs_dim if cfg.coarse_obs == "obs" else cfg.determ_dim, 
            self.coarse_dim, 
            **asdict(cfg.coarse_stoch_cfg) if is_dataclass(cfg.coarse_stoch_cfg) else cfg.coarse_stoch_cfg
        )
        
        self.coarse_stoch_dim = self.c_prior.stoch_dim if cfg.coarse_stoch_cfg.stoch_dim else 0

        self.precise_dyn = getattr(rnn, f"{cfg.precise_rnn}Cell")(
            input_dim + self.coarse_stoch_dim + (cfg.stoch_cfg.stoch_dim*cfg.stoch_cfg.n_class),
            cfg.determ_dim,
            obs_dim if cfg.init_from_ == "obs" else cfg.init_from_,
            **asdict(cfg.precise_cfg) if is_dataclass(cfg.precise_cfg) else cfg.precise_cfg,
        )
        self._d_is_vqrnn = "VQ" in cfg.precise_rnn
        self.d_prior = Transition(
            cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg)
        self.d_posterior = Representation(
            obs_dim, cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg
        )


        assert (
            self.d_prior.stoch_dim
            == self.d_posterior.stoch_dim
        ), "stoch_dim must be the same for all stochastic layers"

        self.stoch_dim = self.d_prior.stoch_dim if cfg.stoch_cfg.stoch_dim else 0
        self.determ_dim = cfg.determ_dim
        self.latent_dim = cfg.determ_dim + self.stoch_dim + self.coarse_dim + self.coarse_stoch_dim
        self.coarse_latent_dim = self.coarse_dim + self.coarse_stoch_dim
        self.use_stoch = "posterior"
        self.latent_dim_for_action = self.latent_dim
        self.coarse_obs = cfg.coarse_obs


    def init_latent(self, batch_size, obs=None):
        self.coarse_state = self.coarse_dyn.init_latent(
            obs if self.init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)

        self.hidden_state = self.precise_dyn.init_latent(
            obs if self.init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)
        if self.init_with_ == "posterior":
            c_posterior = self.c_posterior(
                    self.coarse_state, 
                    obs if self.cfg.coarse_obs == "obs" else self.hidden_state)
            self.prev_c_stoch = c_posterior.stoch
            # d_posterior = self.d_posterior(
            #         self.hidden_state, 
            #         obs) 
            d_posterior = self.posterior(
                    self.hidden_state, 
                    obs) 
            self.prev_stoch = d_posterior.stoch
        elif self.stoch_dim:
            self.prev_stoch = self.d_prior(self.hidden_state).stoch
            self.prev_c_stoch = self.c_prior(self.coarse_state).stoch
        else:
            self.prev_stoch = None
        return torch.cat([self.hidden_state, self.prev_stoch, self.coarse_state, self.prev_c_stoch], dim=-1)

    def set_prev_states(self, worlds: Worlds, set_mtrnn_hidden=True):
        self.hidden_state = worlds.determ
        self.prev_stoch = worlds.posterior.stoch
        self.coarse_state = worlds.coarse
        self.prev_c_stoch = worlds.c_posterior.stoch
        # print(self.prev_stoch.shape)
        # print(self.prev_c_stoch.shape)
        if isinstance(self.precise_dyn, rnn.MTRNNCell) and set_mtrnn_hidden:
            if self.hidden_state.shape[0] == 1:
                self.precise_dyn.hidden = torch.cat(self.precise_dyn.hidden_histopy)[-1].unsqueeze(0).expand(
                    self.hidden_state.shape[0], -1, -1).flatten(0, 1)
            else:
                self.precise_dyn.hidden = torch.cat(self.precise_dyn.hidden_histopy).unsqueeze(0).expand(
                        self.hidden_state.shape[0]//len(self.precise_dyn.hidden_histopy), -1, -1).flatten(0, 1)
        if isinstance(self.coarse_dyn, rnn.MTRNNCell) and set_mtrnn_hidden:
            if self.coarse_state.shape[0] == 1:
                self.coarse_dyn.hidden = torch.cat(self.coarse_dyn.hidden_histopy)[-1].unsqueeze(0).expand(
                    self.coarse_state.shape[0], -1, -1).flatten(0, 1)
                # print(self.coarse_dyn.hidden.shape)
            else:
                self.coarse_dyn.hidden = torch.cat(self.coarse_dyn.hidden_histopy).unsqueeze(0).expand(
                    self.coarse_state.shape[0]//len(self.coarse_dyn.hidden_histopy), -1, -1).flatten(0, 1)
            
        return torch.cat([self.hidden_state, self.prev_stoch, self.coarse_state, self.prev_c_stoch], dim=-1)

    def detach(self):
        self.hidden_state = self.hidden_state.detach()
        self.prev_stoch = self.prev_stoch.detach()
        self.coarse_state = self.coarse_state.detach()
        self.prev_c_stoch = self.prev_c_stoch.detach()
        if hasattr(self.coarse_dyn, "hidden_state"):
            self.coarse_dyn.hidden_state = self.coarse_dyn.hidden_state.detach()
        self.precise_dyn.detach()
        self.coarse_dyn.detach()

    
    def step(self, action: torch.Tensor, obs: torch.Tensor =None, deterministic:bool = False) -> Tuple[CoarseWorldStates, Dict[str, torch.Tensor]]:
        loss_dict = {}
        # print(self.prev_c_stoch.shape)
        # print(self.coarse_state.shape)
        coarse_returns = self.coarse_dyn(self.prev_c_stoch, self.coarse_state)
        if self._c_is_vqrnn:
            self.coarse_state, c_vq_loss, gate = coarse_returns
            loss_dict["c_vq_loss"] = c_vq_loss
        else:
            self.coarse_state, gate = coarse_returns
        c_prior = self.c_prior(
            self.coarse_state, deterministic=deterministic)
        c_posterior = self.c_posterior(
            self.coarse_state, 
            obs if self.cfg.coarse_obs == "obs" else self.hidden_state, 
            deterministic=deterministic
        ) if obs is not None or self.cfg.coarse_obs == "determ" else c_prior
        self.prev_c_stoch = c_posterior.stoch

        # print(action.shape)
        determ_state = self.precise_dyn(
            torch.cat([action, self.prev_stoch, self.prev_c_stoch], dim=-1), self.hidden_state
        )
        if self._d_is_vqrnn:
            self.hidden_state, d_vq_loss = determ_state
            loss_dict["d_vq_loss"] = d_vq_loss
        else:
            self.hidden_state = determ_state

        d_prior = self.d_prior(
            self.hidden_state, deterministic=deterministic)
        d_posterior = self.d_posterior(
            self.hidden_state, obs, deterministic=deterministic) if obs is not None else d_prior
        states = CoarseWorldStates(self.hidden_state, self.coarse_state,
                             d_prior, d_posterior, c_prior, c_posterior, gate)

        self.prev_stoch = states.posterior.stoch if obs is not None else states.prior.stoch
        
        # print(self.hidden_state.shape)
        # print(self.prev_stoch.shape)

        return states, loss_dict

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor, deterministic:bool = False) -> Tuple[Worlds, Dict[str, torch.Tensor]]:
        """
        Args:
            action: shape(T, B, D)
            embed_obs: shape(T, B, D)

        """

        world_history = []
        loss_history = []

        for t in range(len(action)):
            world_states, loss_dict = self.step(action[t], embed_obs[t], deterministic)
            world_history.append(world_states)
            loss_history.append(loss_dict)
        world_history = stack_worlds(world_history)
        if self._c_is_vqrnn or self._d_is_vqrnn:
            loss_history = stack_dicts(loss_history)
        else:
            loss_history = None

        return world_history, loss_history
