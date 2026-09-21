import torch
from typing import Dict, Tuple
from utils.states import (CategoricStoch, NormalStoch, WorldStates, WorldStatesLayer,
                              CoarseWorldStates, Worlds, stack_worlds, stack_dicts)
from utils.config import MTRSSMConfig, RSSMConfig,CRSSMConfig, CRSSMV4Config
from dataclasses import asdict, is_dataclass
from utils.utils import mytorch
import torch.nn as nn
from networks.distributions import Representation, Transition
import networks.rnn as rnn

class RSSMPredictor(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg: RSSMConfig):
        """"
         入力:

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
        引数:
            action[B,T,action_dim]: 時刻tからt+1への遷移。
            observation[B,T,obs_dim]: 時刻t+1における行動後の観測。
            state[B,latent_dim]: 最初の行動前の時刻tにおける状態。
            initial_obs[B,obs_dim]: 引き継がれた状態がない場合に、最初の状態を
                推論するために使用する観測。
        
        戻り値:
            outputs[B,T,obs_dim]: 行動前の各時刻における再構成結果。
            hidden_all[B,T,latent_dim]: 各出力と同じ時刻に対応する状態。
            hidden_last[B,latent_dim]: 次へ引き継ぐ、最後の行動後の状態。
        """

        batch_size = action.shape[0]
        # デコーダーはaction[t]を適用する前の現在の状態から観測を再構成する。
        # その状態は、以下のT回の遷移によって生成されるT個の状態とは分けて保持する。
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

            # stateを先頭からdetrm_dim個までをhidden_stateに、残りをprev_stochに分割してRSSMの状態として設定する。
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
        # worlds:各時刻でRSSMが計算した潜在状態をまとめたもの
        # aux_loss:VQ-RNNを使用している場合の補助損失
        worlds,aux_loss=self.rssm(
            action=action_tbd,
            embed_obs=embed_obs_tbd,
        )

        # 各遷移後の状態は、action[t]を適用した後の観測に対応する:
        # [state_(t+1), ..., state_(t+T)].
        next_latent_tbd = torch.cat(
            [
                worlds.determ,
                worlds.posterior.stoch,
            ],
            dim=-1,
        )

        # next_latent_tbd[state_t, ..., state_(t+T-1)]と行動を実行する直前の潜在状態列から行動前の観測を再構成する。
        # 最後の遷移後の状態は、次のBPTTウィンドウへ引き継ぐ再帰状態としてのみ保持する。
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
    """決定論的状態と確率的状態を組み合わせた状態空間モデル。

    各時刻の潜在状態は、RNNが保持する決定論的状態 ``h_t`` と、
    確率分布から得る確率的状態 ``z_t`` で構成される。

    時刻 ``t`` から ``t+1`` への更新は、概念的には次のとおり。

        h_(t+1) = RNN(h_t, [action_t, z_t])
        prior     = p(z_(t+1) | h_(t+1))
        posterior = q(z_(t+1) | h_(t+1), obs_(t+1))

    観測がある場合はposteriorのサンプルを、観測がない将来予測では
    priorのサンプルを次の時刻の ``z`` として使用する。

    記号:
        B: バッチサイズ
        T: 系列長
        H: 決定論的状態の次元 ``determ_dim``
        Z: 確率的状態の次元 ``stoch_dim``
    """
    def __init__(
        self,
        obs_dim: int,
        input_dim: int,
        cfg: RSSMConfig
    ):
        super().__init__()
        # 他の階層型RSSMと共通のインターフェースを保つための属性。
        # 単層のRSSMでは粗い時間スケールの観測を使用しない。
        self.coarse_obs = None

        # 決定論的状態hを更新する再帰セルを設定する。
        # セルへの入力は、行動と前時刻の確率的状態zを結合したもの。
        # カテゴリカル分布の場合、zは
        # stoch_dim * n_class次元のベクトルとして平坦化されている。
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
            # priorは観測を使わず、決定論的状態hだけから
            # 確率的状態zの予測分布p(z|h)を計算する。
            self.prior = Transition(
                cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg)
            # posteriorはhと実際の観測obsから、
            # zの事後分布q(z|h, obs)を計算する。
            self.posterior = Representation(
                obs_dim, cfg.determ_dim, **asdict(cfg.stoch_cfg) if is_dataclass(cfg.stoch_cfg) else cfg.stoch_cfg
            )
            # priorとposteriorは同じ形の確率的状態を表す必要がある。
            assert self.prior.stoch_dim == self.posterior.stoch_dim

        # RSSMの完全な潜在状態は[h, z]を最終次元で結合したもの。
        self.stoch_dim = self.prior.stoch_dim if cfg.stoch_cfg.stoch_dim else 0
        self.determ_dim = cfg.determ_dim
        self.latent_dim = cfg.determ_dim + self.stoch_dim
        self.use_stoch = "posterior"
        self.latent_dim_for_action = self.latent_dim

    def init_latent(self, batch_size, obs=None):
        """系列の開始時にRSSM内部の状態を初期化する。

        引数:
            batch_size: バッチサイズB。
            obs: 初期観測。``init_from_ == "obs"`` の場合はhの初期化に、
                ``init_with_ == "posterior"`` の場合はzの初期化にも使う。

        戻り値:
            ``[h_0, z_0]`` を結合した形状 ``[B, H+Z]`` の潜在状態。
        """
        # 設定に応じて、初期観測、ゼロ、または学習可能なパラメータから
        # 最初の決定論的状態h_0を作る。
        self.hidden_state = self.rnn.init_latent(
            obs if self._init_from_ == "obs" else batch_size
        ).reshape(batch_size, -1)

        if self._init_with_ == "posterior":
            # 初期観測がある場合、観測を反映したposteriorからz_0を得る。
            self.prev_stoch = self.posterior.forward(
                self.hidden_state, obs).stoch.reshape([batch_size, -1])
        elif self.stoch_dim > 0:
            # 観測を使わない設定では、h_0から計算したpriorからz_0を得る。
            self.prev_stoch = self.prior(self.hidden_state).stoch.reshape([batch_size, -1])
        else:
            self.prev_stoch = None

        # 呼び出し側で一つの状態として引き継げるよう[h_0, z_0]にまとめる。
        return mytorch.concat([self.hidden_state, self.prev_stoch], dim=-1)

    def set_prev_states(self, worlds: Worlds):
        """WorldStatesを次のステップで使用する内部状態として設定する。

        観測済みの状態を引き継ぐため、確率的状態にはposteriorの
        サンプルを使用する。
        """
        self.hidden_state = worlds.determ
        self.prev_stoch = worlds.posterior.stoch
        return torch.cat([self.hidden_state, self.prev_stoch], dim=-1)

    def detach(self):
        """打ち切りBPTTのチャンク間で内部状態を計算グラフから切り離す。

        値は保持したまま過去のチャンクへの勾配伝播を止めることで、
        計算グラフが系列全体へ無制限に伸びることを防ぐ。
        """
        if hasattr(self, "hidden_state") and self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()
        if hasattr(self, "prev_stoch") and self.prev_stoch is not None:
            self.prev_stoch = self.prev_stoch.detach()
        if hasattr(self.rnn, "detach"):
            self.rnn.detach()

    def step(self, action, obs=None, timestep: int = 0) -> WorldStates:
        """行動を1ステップ適用し、次時刻の潜在状態を計算する。

        呼び出し時の内部状態を ``[h_t, z_t]`` とすると、``action_t`` と
        行動後の観測 ``obs_(t+1)`` から ``[h_(t+1), z_(t+1)]`` を作る。

        引数:
            action: 時刻tからt+1へ遷移する行動。形状 ``[B, action_dim]``。
            obs: 行動後の観測 ``obs_(t+1)``。形状 ``[B, obs_dim]``。
                ``None`` の場合は観測なしの予測としてpriorを使用する。
            timestep: 呼び出し元における時刻番号。現在の単層RSSMでは
                状態更新の計算には使用しない。

        戻り値:
            次時刻の決定論的状態、prior、posteriorを格納した
            ``WorldStates`` と、VQ-RNN使用時の補助損失。
        """
        # 1. 前状態[h_t, z_t]とaction_tから、次の決定論的状態h_(t+1)を得る。
        determ_state = self.rnn(mytorch.concat(
            [action, self.prev_stoch], dim=-1), self.hidden_state)

        # VQ-RNNは状態に加えて量子化の補助損失を返す想定。
        # 通常のRNN/GRUではdeterm_stateそのものが次のhidden_stateになる。
        if self._is_vqrnn:
            self.hidden_state, vq_loss = determ_state
            loss_dict = {"vq_loss": vq_loss}
        else:
            self.hidden_state = determ_state
            loss_dict = None

        if self.stoch_dim:
            # 2. 観測を使わずに、h_(t+1)から予測分布priorを計算する。
            prior = self.prior(determ_state)
            # 3. 観測がある場合はh_(t+1)とobs_(t+1)からposteriorを計算する。
            #    観測がなければposteriorをpriorと同一にして将来予測を行う。
            posterior = self.posterior(
                determ_state, obs) if obs is not None else prior
        else:
            prior = NormalStoch()
            posterior = NormalStoch()

        # 同じ時刻t+1の決定論的状態と二つの確率分布を一つにまとめる。
        states = WorldStates(determ_state, prior, posterior)

        # 4. 次の更新で使うz_(t+1)を保存する。
        #    観測時はposterior、観測なしの予測時はpriorのサンプルを使う。
        self.prev_stoch = states.posterior.stoch if obs is not None else states.prior.stoch

        return states, loss_dict

    def forward(self, action: torch.Tensor, embed_obs: torch.Tensor):
        """系列全体を先頭から順に処理する。

        引数:
            action: 各時刻の行動。形状 ``[T, B, action_dim]``。
            embed_obs: 各行動後の観測。形状 ``[T, B, obs_dim]``。

        戻り値:
            world_history: 各時刻の``WorldStates``を時間方向に積んだもの。
                ``determ``は ``[T, B, H]``、prior/posteriorのサンプルは
                ``[T, B, Z]`` となる。
            loss_history: VQ-RNNの場合は時刻ごとの補助損失を積んだ辞書。
                通常のRNN/GRUの場合は ``None``。

        時間方向の間引きは行わず、入力されたTステップをすべて処理する。
        """

        world_history = []
        loss_history = []

        # action[t]はtからt+1への行動、embed_obs[t]は行動後の観測。
        # 各時刻のstep結果は、対応する次時刻t+1の状態になる。
        for t in range(len(action)):
            world_states, loss = self.step(action[t], embed_obs[t], t)
            world_history.append(world_states)
            loss_history.append(loss)

        # 時刻ごとのオブジェクトを、先頭に時間次元Tを持つ形へまとめる。
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
        引数:
            action: 形状(T, B, D)
            embed_obs: 形状(T, B, D)

        """

        world_history = []

        for t in range(len(action)):
            world_states = self.step(action[t], embed_obs[t], t)

            world_history.append(world_states)

        world_history = stack_worlds(world_history)

        return world_history


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
        引数:
            action: 形状(T, B, D)
            embed_obs: 形状(T, B, D)

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
        引数:
            action: 形状(T, B, D)
            embed_obs: 形状(T, B, D)

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
