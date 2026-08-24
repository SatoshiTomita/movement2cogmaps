from dataclasses import asdict, is_dataclass
from typing import Literal, Tuple, Union

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from networks.activations import Activation
from networks.base import LightningModuleBase
from networks.dynamics import MTRSSM, RSSM, CRSSMV4, CRSSM

from utils.states import cat_dicts, sum_dicts
from networks.layers import (MLPLayer, States2Map1d, 
                                 States2Map2d, SoftmaxTransformation,
                                 Map2States)
from networks.vision import Decoder, Encoder
from utils.config import (
    CRSSMV4Config,
    MTRSSMConfig,
    CRSSMConfig,
    RSSMConfig,
    WorldConfig,
)

from utils.loss import CalcFreeEnergy
from utils.loss import LossFunctions as lf
from utils.states import StochState, CoarseWorldStates, Worlds, get_dist
from utils.utils import mytorch, HeavisideStepFnc


class WorldModel(LightningModuleBase):
    def __init__(self, cfg: WorldConfig):
        self.optimizer_cfg = cfg.optimizer_cfg
        assert cfg.alpha >= 0 or cfg.alpha <= 1, "alpha must be from 0 to 1"
        super().__init__()
        self.cfg = cfg
        self.alpha = cfg.alpha


        if cfg.action_transformation is not None:
            self.action_transform = SoftmaxTransformation(cfg.action_transformation)
            action_dim = self.action_transform.get_transformed_dim(cfg.action_dim)
        else:
            self.action_transform = nn.Identity()
            action_dim = cfg.action_dim
        self.loss_cfg = OmegaConf.create(cfg.loss_cfg)

        self.contrastive_class = cfg.contrastive_class
        self.contrastive_activation = cfg.contrastive_activation
        self.embed_obs_dim = cfg.obs_dim

        self.obs_encoder = Encoder(cfg.obs_shape, cfg.obs_dim, cfg.encoder_cfg)
        if isinstance(cfg.dynamics_cfg, RSSMConfig):
            self.dynamics = RSSM(cfg.obs_dim, action_dim, cfg.dynamics_cfg)
        elif isinstance(cfg.dynamics_cfg, MTRSSMConfig):
            self.dynamics = MTRSSM(cfg.obs_dim, action_dim, cfg.dynamics_cfg)
        elif isinstance(cfg.dynamics_cfg, CRSSMConfig):
            self.dynamics = CRSSM(cfg.obs_dim, action_dim, cfg.dynamics_cfg)
        elif isinstance(cfg.dynamics_cfg, CRSSMV4Config):
            self.dynamics = CRSSMV4(cfg.obs_dim, action_dim, cfg.dynamics_cfg)
        else:
            raise NotImplementedError

        if self.alpha > 0 and not cfg.hypernetize:
            if self.loss_cfg.embed_beta:
                self.embed_decoder = nn.Sequential(
                    nn.Linear(self.dynamics.latent_dim, self.obs_encoder.embed_dim),
                    Activation(cfg.activation),
                    nn.Linear(self.obs_encoder.embed_dim, self.obs_encoder.embed_dim),
                    self.obs_encoder.encoder.output_activation,
                )

                self.obs_decoder = Decoder(cfg.obs_dim, cfg.obs_shape, cfg.decoder_cfg)

            else:
                self.obs_decoder = Decoder(
                    self.dynamics.latent_dim, cfg.obs_shape, cfg.decoder_cfg
                )

        if self.alpha < 1:
            self.critic_obs = nn.Sequential(
                nn.Linear(self.obs_encoder.embed_dim, cfg.contrastive_hidden),
                Activation(cfg.activation),
                nn.Linear(cfg.contrastive_hidden, self.contrastive_class),
            )

            self.critic_latents = nn.Sequential(
                nn.Linear(self.dynamics.latent_dim, cfg.contrastive_hidden),
                Activation(cfg.activation),
                nn.Linear(cfg.contrastive_hidden, self.contrastive_class),
            )
            self.critic_activation = Activation(**cfg.contrastive_activation)

        else:
            self.critic_obs = None
            self.critic_latents = None
            self.critic_activation = None

        self.spatial = getattr(cfg.encoder_cfg, "spatial", False)
        if self.cfg.truncated:
            self.truncated = self.cfg.truncated
            self.automatic_optimization = False
        else:
            self.truncated = False

    def _shared_step(self, batch):
        if self.truncated:
            act_in, obs_in, obs_target = batch
            trunc_act = act_in.split(self.truncated, dim=0)
            trunc_obs = obs_in.split(self.truncated, dim=0)
            trunc_target = obs_target.split(self.truncated, dim=0)
            opt = self.optimizers()
            predicts = []
            loss_history = []
            for i in range(len(trunc_act)):
                wld_loss, embed_obs, pred_dict = self._train(
                    (trunc_act[i], trunc_obs[i], trunc_target[i]), i==0
                )
                if self.training:
                    opt.zero_grad()
                    self.manual_backward(wld_loss["loss"])
                    opt.step()
                    self.dynamics.detach()
                predicts.append(pred_dict)
                loss_history.append(wld_loss)
            return sum_dicts(loss_history), cat_dicts(predicts)

        else:
            wld_loss, embed_obs, pred_dict = self._train(batch)
            return wld_loss, pred_dict

    def _current_latent(self) -> torch.Tensor:
        """Return the recurrent state before the next action is applied."""
        if isinstance(self.dynamics, MTRSSM):
            return torch.cat(
                [
                    self.dynamics.low_level.hidden_state,
                    self.dynamics.low_level.prev_stoch,
                    self.dynamics.high_level.hidden_state,
                    self.dynamics.high_level.prev_stoch,
                ],
                dim=-1,
            )
        if isinstance(self.dynamics, (CRSSM, CRSSMV4)):
            states = [
                self.dynamics.hidden_state,
                self.dynamics.prev_stoch,
                self.dynamics.coarse_state,
            ]
            if isinstance(self.dynamics, CRSSMV4):
                states.append(self.dynamics.prev_c_stoch)
            return torch.cat(states, dim=-1)
        return torch.cat(
            [self.dynamics.hidden_state, self.dynamics.prev_stoch], dim=-1
        )

    def rollout(
        self,
        act_in: torch.Tensor,
        obs_in: torch.Tensor,
        init: bool = True,
        initial_obs: torch.Tensor = None,
    ) -> Tuple[Worlds, torch.Tensor, dict]:
        """Infer states aligned with ``obs_in`` after applying ``act_in``.

        ``act_in[t]`` represents the transition from time ``t`` to ``t+1``
        and ``obs_in[t]`` is the post-action observation at ``t+1``.  At the
        start of a sequence, ``initial_obs`` supplies the pre-action
        observation at time ``t`` used to initialize the recurrent state.

        The optional argument keeps the older three-item WorldModel batch API
        working for callers whose observations are not transition-aligned.
        """
        embed_obs = self.obs_encoder(obs_in)

        if init:
            initial_embed = (
                self.obs_encoder(initial_obs)
                if initial_obs is not None
                else embed_obs[0]
            )
            self.dynamics.init_latent(embed_obs.shape[1], initial_embed)

        # Keep the pre-action state for activity analyses, whose positions and
        # headings are indexed by the current observations rather than labels.
        self._rollout_start_latent = self._current_latent()
        world_states, loss_dict = self.dynamics.forward(act_in, embed_obs)
        return world_states, embed_obs, loss_dict

    def _train(self, batch: tuple, init: bool = True):

        if len(batch) == 3:
            act_in, obs_in, obs_target = batch
            initial_obs = None
        elif len(batch) == 4:
            act_in, obs_in, obs_target, initial_obs = batch
        else:
            raise ValueError(
                "WorldModel batches must contain action, observation, target, "
                "and optionally the initial pre-action observation"
            )

        world_states, embed_obs, loss_dict = self.rollout(
            act_in,
            obs_in,
            init,
            initial_obs=initial_obs,
        )

        latent_states = world_states.latent_states 
        current_latent_states = torch.cat(
            [self._rollout_start_latent.unsqueeze(0), latent_states[:-1]],
            dim=0,
        )

        # Reconstruct o_t from the state at t.  The transition states returned
        # above are [s_(t+1), ..., s_(t+T)], so prepend the pre-action state and
        # drop the final transition state.  MTRSSM image reconstruction uses
        # only the lower-level h/z portion of that aligned state history.
        if isinstance(self.dynamics, MTRSSM):
            lower_dim = self.dynamics.latent_dim
            decoder_states = torch.cat(
                [
                    self._rollout_start_latent[..., :lower_dim].unsqueeze(0),
                    world_states.layer0.latent_states[:-1],
                ],
                dim=0,
            )
        else:
            decoder_states = current_latent_states
        predicted_obs, predicted_embed_obs = self._decode_obs(decoder_states, embed_obs)

        if self.obs_decoder.decode_edge:
            predicted_obs, predicted_edge = predicted_obs.split([3, 6], dim=-3)


        wld_loss = CalcFreeEnergy.variational_alpha_sub(
            world_states.posterior,
            world_states.prior,
            obs_target,
            predicted_obs,
            embed_obs,
            latent_states,
            self.critic_latents,
            self.critic_obs,
            self.critic_activation,
            self.alpha,
            **self.loss_cfg
        )
        if loss_dict is not None:
            for key in loss_dict.keys():
                wld_loss["loss"] += loss_dict[key].mean()
                wld_loss[key] = loss_dict[key].detach().mean().item()


        if isinstance(self.dynamics, MTRSSM):
            top_complexity = (
                lf.kl_balancing(
                    world_states.layer1.posterior,
                    world_states.layer1.prior,
                    self.loss_cfg.kl_balancing,
                )
                if self.loss_cfg.kl_balancing
                else lf.kl_vanilla(
                    world_states.layer1.posterior,
                    world_states.layer1.prior,
                )
            )

            top_complexity = torch.max(
                top_complexity,
                top_complexity.new_full(top_complexity.size(), self.loss_cfg.free_nats),
            )
            wld_loss["loss"] = wld_loss["loss"] + top_complexity
            wld_loss["top_complexity"] = top_complexity.item()



        pred_dict = dict(
            prediction=predicted_obs,
            predicted_embed_obs=predicted_embed_obs,
            latent_states=latent_states,
            analysis_latent_states=current_latent_states,
            world_states=world_states,
        )

        return wld_loss, embed_obs, pred_dict  

    def _decode_obs(
        self, latent_states: torch.Tensor, *args
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
        if self.loss_cfg.embed_beta:
            predicted_embed_obs = self.embed_decoder(latent_states)

            predicted_obs = self.obs_decoder(predicted_embed_obs)

            return predicted_obs, predicted_embed_obs

        elif self.alpha > 0:
            predicted_obs = self.obs_decoder(latent_states)
            return predicted_obs, None
        else:
            return None, None

class CoarseWorldModel(WorldModel):
    def __init__(
            self,
            cfg: WorldConfig,
            ):
        super().__init__(cfg)

        self.heaviside_fnc = HeavisideStepFnc.apply
        if cfg.alpha < 1:
            self.critic_c_latents = nn.Sequential(
                nn.Linear(self.dynamics.coarse_latent_dim, cfg.contrastive_hidden),
                Activation(cfg.activation),
                nn.Linear(cfg.contrastive_hidden, self.contrastive_class),
            )
            if not cfg.shared_critic:
                self.critic_c_obs = nn.Sequential(
                    nn.Linear(
                        self.obs_encoder.embed_dim if self.dynamics.coarse_obs == "obs" else self.dynamics.determ_dim + self.dynamics.stoch_dim, 
                        cfg.contrastive_hidden),
                    Activation(cfg.activation),
                    nn.Linear(cfg.contrastive_hidden, self.contrastive_class),
                )
        else:
            self.critic_c_latents = None
        self.coarse_obs = self.dynamics.coarse_obs
        if cfg.alpha > 0:
            if cfg.shared_decoder and (isinstance(self.dynamics, CRSSMV4) or isinstance(self.dynamics, CRSSM)):
                if self.dynamics.cfg.coarse_obs == "determ" and self.loss_cfg.w_coarse_obs > 0:
                    self.coarse_obs_decoder = MLPLayer(
                        self.dynamics.coarse_latent_dim, 
                        self.dynamics.determ_dim, 
                        cfg.c_decoder_cfg
                    )
                else:
                    self.obs_decoder.make_specific_projection(self.dynamics.coarse_latent_dim)
            else:
                self.coarse_obs_decoder = Decoder(
                    self.dynamics.coarse_latent_dim, cfg.obs_shape, cfg.decoder_cfg
                )
    def decode_coarse_obs(
            self, 
            coarse_latent_states: torch.Tensor, 
            posterior_stoch: torch.Tensor = None,
            return_prior: bool = False,
            n_samples: int = 1
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.shared_decoder:
            if self.coarse_obs == "determ":
                if self.loss_cfg.w_coarse_obs > 0:
                    predicted_determ = self.coarse_obs_decoder(coarse_latent_states)
                    d_prior = self.dynamics.d_prior(predicted_determ)
                    if n_samples > 1:
                        d_prior_dist = get_dist(d_prior)
                        samples = d_prior_dist.rsample([n_samples]).reshape(
                            n_samples**2, 
                            -1,
                            d_prior.stoch.shape[-1]
                        )
                        predicted_determ = predicted_determ.unsqueeze(0).expand(n_samples, -1, -1, -1).flatten(0, 1)
                        coarse_latent_states = coarse_latent_states.unsqueeze(0).expand(n_samples, -1, -1, -1).flatten(0, 1)
                        
    
                    else:
                        samples = d_prior.stoch
                else:
                    predicted_determ = None
                    d_prior = None
                    samples = None
    
                if self.loss_cfg.recon_coarse:
                    latent_states = mytorch.concat(
                            [predicted_determ, 
                             posterior_stoch if posterior_stoch is not None else samples,
                             coarse_latent_states], dim=-1
                            )
                    
                    coarse_predicted_obs = self.obs_decoder(latent_states)
                else:
                    coarse_predicted_obs = None
                if return_prior:
                    return coarse_predicted_obs, predicted_determ, d_prior
                else:
                    return coarse_predicted_obs, predicted_determ
            else:
                coarse_predicted_obs = self.obs_decoder.decode_from_specific_projection(coarse_latent_states)
                if return_prior:
                    return coarse_predicted_obs, None, None
                else:
                    return coarse_predicted_obs, None
        else:
            coarse_predicted_obs = self.coarse_obs_decoder(coarse_latent_states)
            if return_prior:
                return coarse_predicted_obs, None, None
            else:
                return coarse_predicted_obs, None

    def _shared_step(self, batch):
        if self.truncated:
            act_in, obs_in, obs_target = batch
            trunc_act = act_in.split(self.truncated, dim=0)
            trunc_obs = obs_in.split(self.truncated, dim=0)
            trunc_target = obs_target.split(self.truncated, dim=0)
            opt = self.optimizers()
            predicts = []
            loss_history = []
            for i in range(len(trunc_act)):
                wld_loss, embed_obs, pred_dict = self._train(
                    (trunc_act[i], trunc_obs[i], trunc_target[i]), i==0
                )
                wld_loss, pred_dict = self._coarse_train(wld_loss, embed_obs, pred_dict, trunc_target[i])
                if self.training:
                    opt.zero_grad()
                    self.manual_backward(wld_loss["loss"])
                    opt.step()
                    self.dynamics.detach()
                predicts.append(pred_dict)
                loss_history.append(wld_loss)
            return sum_dicts(loss_history), cat_dicts(predicts)

        else:
            wld_loss, embed_obs, pred_dict = self._train(batch)
            return self._coarse_train(wld_loss, embed_obs, pred_dict, batch[-1])

    def _coarse_train(self, wld_loss, embed_obs, pred_dict, obs_target):
        world_states = pred_dict.pop("world_states")
        world_states: CoarseWorldStates = world_states
        coarse_latent_states = world_states.coarse_states

        coarse_predicted_obs, predicted_determ = self.decode_coarse_obs(coarse_latent_states)
        if self.obs_decoder.decode_edge:
            coarse_predicted_obs, coarse_predicted_edge = coarse_predicted_obs.split([3, 6], dim=-3)
        if self.loss_cfg.recon_coarse:
            pred_dict["coarse_prediction"] = coarse_predicted_obs

            c_wld_loss = CalcFreeEnergy.variational_alpha_sub(
                world_states.c_posterior if isinstance(self.dynamics, CRSSMV4) else world_states.posterior,
                world_states.c_prior,
                obs_target,
                coarse_predicted_obs,
                embed_obs if self.dynamics.coarse_obs == "obs" or self.cfg.shared_critic else torch.cat([world_states.determ, world_states.posterior.stoch], dim=-1),
                coarse_latent_states,
                self.critic_c_latents,
                self.critic_obs if self.cfg.shared_critic else self.critic_c_obs,
                self.critic_activation,
                self.alpha,
                **self.loss_cfg
            )
        else:
            coarse_complexity = (
                lf.kl_balancing(
                    world_states.c_posterior if isinstance(self.dynamics, CRSSMV4) else world_states.posterior,
                    world_states.c_prior,
                    self.loss_cfg.kl_balancing,
                )
                if self.loss_cfg.kl_balancing
                else lf.kl_vanilla(
                    world_states.c_posterior if isinstance(self.dynamics, CRSSMV4) else world_states.posterior,
                    world_states.c_prior,
                ))
            coarse_complexity = torch.max(
                coarse_complexity,
                coarse_complexity.new_full(coarse_complexity.size(), self.loss_cfg.free_nats),
                )
            c_wld_loss = dict(
                loss=coarse_complexity * self.loss_cfg.kl_beta,
                complexity=coarse_complexity.item(),
                )
        wld_loss["loss"] += c_wld_loss.pop("loss")
        for c_wld_loss_key in c_wld_loss.keys():
            wld_loss[f"coarse_{c_wld_loss_key}"] = c_wld_loss[c_wld_loss_key]


        if self.loss_cfg.w_coarse_obs > 0 and self.coarse_obs == "determ" and self.cfg.shared_decoder:
            coarse_obs_loss = F.mse_loss(predicted_determ, world_states.determ) * self.dynamics.determ_dim 
            wld_loss["coarse_obs_loss"] = coarse_obs_loss.item()
            wld_loss["loss"] += coarse_obs_loss * self.loss_cfg.w_coarse_obs

        if world_states.gate is not None:
            coarse_l0_norm = self.heaviside_fnc(world_states.gate).mean() * self.loss_cfg.w_l0_norm
            wld_loss["loss"] += coarse_l0_norm
            wld_loss["coarse_l0_norm"] = coarse_l0_norm.item()

        return wld_loss, pred_dict
