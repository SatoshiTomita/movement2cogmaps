import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from networks.activations import Activation, TanhandREReLU
from utils.states import StochState, get_dist


class LossFunctions:
    @staticmethod
    def charbonnier(
        prediction: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-3,
        alpha: float = 1,
        sum_dim: int = [-1, -2, -3],
    ) -> torch.Tensor:
        x = prediction - target
        loss = (x**2 + epsilon**2) ** (alpha / 2)
        return torch.sum(loss, dim=sum_dim).mean()
    
    @staticmethod
    def charbonnier_std(
        prediction: torch.Tensor,
        target: torch.Tensor,
        epsilon: float = 1e-3,
        alpha: float = 1,
        sum_dim: int = [-1, -2, -3],
    ) -> torch.Tensor:
        x = prediction - target
        loss = (x**2 + epsilon**2) ** (alpha / 2)
        mean_dim = tuple(np.arange(loss.ndim - len(sum_dim)))
        std = torch.std(loss, dim=mean_dim)
        loss = loss * std
        return torch.sum(loss, dim=sum_dim).mean()

    @staticmethod
    def mse_std(prediction: torch.Tensor, target: torch.Tensor, sum_dim: int = -1) -> torch.Tensor:
        mse = F.mse_loss(prediction, target, reduction="none")
        mean_dim = tuple(np.arange(mse.ndim - len(sum_dim)))
        std = torch.std(mse, dim=mean_dim)
        mse = mse * std
        return torch.sum(mse, dim=sum_dim).mean()
    
    @staticmethod
    def mse(prediction: torch.Tensor, target: torch.Tensor, sum_dim: int = -1) -> torch.Tensor:
        return F.mse_loss(prediction, target, reduction="none").sum(dim=sum_dim).mean()

    @staticmethod
    def mae(prediction: torch.Tensor, target: torch.Tensor, sum_dim: int = -1) -> torch.Tensor:
        return F.l1_loss(prediction, target, reduction="none").sum(dim=sum_dim).mean()

    @staticmethod
    def kl_vanilla(posterior: StochState, prior: StochState, **kwargs):
        kld = D.kl_divergence(get_dist(posterior), get_dist(prior))
        kld = kld.mean()

        return kld

    @staticmethod
    def kl_balancing(posterior: StochState, prior: StochState, alpha=0.8):
        kld_prior = alpha * D.kl_divergence(
            get_dist(posterior.detach()), get_dist(prior)
        )

        kld_posterior = (1 - alpha) * D.kl_divergence(
            get_dist(posterior), get_dist(prior.detach())
        )

        kld = kld_prior + kld_posterior

        kld = kld.mean()

        return kld

    @staticmethod
    def focal_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        sum_dim: int = -1,
    ) -> torch.Tensor:
        prediction = prediction.unsqueeze(1).transpose(sum_dim, 1).squeeze(-1)
        if gamma:
            log_prob = F.log_softmax(prediction, dim=1)
            prob = torch.exp(log_prob)
            loss = F.nll_loss((1 - prob) ** gamma * log_prob, target, reduction="none")
        else:
            loss = F.cross_entropy(prediction, target, reduction="none")
        return loss.mean(0).sum()

    @staticmethod
    def binary_focal_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        sum_dim: int = -1,
    ) -> torch.Tensor:
        if gamma:
            log_probs = F.logsigmoid(prediction)
            neg_log_probs = F.logsigmoid(-prediction)
            probs = torch.sigmoid(prediction)
            focal_weight = torch.where(target == 1, (1 - probs) ** gamma, 
                                   probs ** gamma)
            loss = torch.where(target == 1, -log_probs, -neg_log_probs)
            loss = focal_weight * loss
        else:
            loss = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        return loss.sum(sum_dim).mean()


class CalcFreeEnergy:
    @staticmethod
    def likelihood_variational(
        posterior: StochState,
        prior: StochState,
        obs: torch.Tensor,
        predicted_obs: torch.Tensor,
        kl_balancing: float = 0.8,
        obs_std: float = 1.0,
        kl_beta: float = 0.8,
        free_nats: float = 3.0,
    ):
        accuracy = (
            -D.Independent(D.Normal(predicted_obs, obs_std),
                           3).log_prob(obs).mean()
        )
        if kl_balancing:
            complexity = LossFunctions.kl_balancing(
                posterior, prior, kl_balancing)
        else:
            complexity = LossFunctions.kl_vanilla(posterior, prior)

        complexity = torch.max(
            complexity,
            complexity.new_full(complexity.size(), free_nats),
        )

        return dict(
            loss=accuracy + (complexity * kl_beta),
            accuracy=accuracy.item(),
            complexity=complexity.item(),
        )

    @staticmethod
    def calc_nce(
        obs_embed: torch.Tensor,
        latent_states: torch.Tensor,
        latent_proj: nn.Module,
        obs_proj: nn.Module,
        proj_activation: Activation,
        use_ce: bool = False,
        ):
        hidden_feature = proj_activation(latent_proj(latent_states))
        obs_feature = proj_activation(obs_proj(obs_embed))

        if hidden_feature.ndim == 3:
            hidden_feature = hidden_feature.flatten(0, 1)
        if obs_feature.ndim == 3:
            obs_feature = obs_feature.flatten(0, 1)
        loss_dict = dict()

        if use_ce:

            labels = (
                torch.Tensor(list(range(len(hidden_feature))))
                .long()
                .to(hidden_feature.device)
            )


            sim_matrix = torch.mm(hidden_feature, obs_feature.T)

            nce_loss = F.cross_entropy(sim_matrix, labels, reduction="mean") - np.log(
                len(sim_matrix)
            )
            loss_dict["nce"] = nce_loss

        else:
            positive = torch.sum(hidden_feature * obs_feature, dim=-1)
            loss_dict["positive"] = positive.detach().clone().mean().item()

            sim_matrix = torch.mm(hidden_feature, obs_feature.T)

            negative = torch.logsumexp(
                sim_matrix, dim=-1) - np.log(len(sim_matrix))
            loss_dict["negative"] = negative.detach().clone().mean().item()
            nce_loss = torch.mean(-positive + negative)
            loss_dict["nce"] = nce_loss

        return loss_dict



    

    @staticmethod
    def contrastive_variational(
        posterior: StochState,
        prior: StochState,
        obs_embed: torch.Tensor,
        latent_states: torch.Tensor,
        latent_proj: nn.Module,
        obs_proj: nn.Module,
        proj_activation: Activation,
        nce_balancing: float = 0.5,
        kl_balancing: float = 0.8,
        kl_beta: float = 0.8,
        free_nats: float = 3.0,
    ):
        # latent_states shape: (batch_t * batch_b, latent_dim)
        hidden_feature: torch.Tensor = proj_activation(
            latent_proj(latent_states))
        # obs_embed shape: (batch_t * batch_b, embed_dim)
        obs_feature: torch.Tensor = proj_activation(obs_proj(obs_embed))

        if hidden_feature.ndim == 3:
            hidden_feature = hidden_feature.flatten(0, 1)
        if obs_feature.ndim == 3:
            obs_feature = obs_feature.flatten(0, 1)

        labels = (
            torch.Tensor(list(range(len(hidden_feature))))
            .long()
            .to(latent_states.device)
        )


        sim_matrix = torch.mm(hidden_feature, obs_feature.T)

        nce_loss = F.cross_entropy(sim_matrix, labels, reduction="mean") - np.log(
            len(sim_matrix)
        )

        if kl_balancing:
            complexity = LossFunctions.kl_balancing(
                posterior, prior, kl_balancing)
        else:
            complexity = LossFunctions.kl_vanilla(posterior, prior)
        complexity = torch.max(
            complexity,
            complexity.new_full(complexity.size(), free_nats),
        )
        return dict(
            loss=nce_loss + (complexity * kl_beta),
            nce=nce_loss.item(),
            complexity=complexity.item(),
        )

    @staticmethod
    def variational_alpha_sub(
        posterior: StochState,
        prior: StochState,
        obs: torch.Tensor,
        predicted_obs: torch.Tensor,
        obs_embed: torch.Tensor,
        latent_states: torch.Tensor,
        latent_proj: nn.Module,
        obs_proj: nn.Module,
        proj_activation: Activation,
        alpha: int = 1,
        kl_balancing: float = 0.8,
        obs_std: float = 1.0,
        kl_beta: float = 1.0,
        free_nats: float = 3.0,
        use_ce: bool = True,
        nce_balancing: float = 0.5,
        accuracy_metric: Literal["mse", "logprob", "charbonnier"] = "logprob",
        **kwargs
    ):
        loss_dict = dict(loss=0)
        accuracy = torch.zeros(1)
        nce_loss = torch.zeros(1)
        if alpha > 0:
            if accuracy_metric == "logprob":
                accuracy = (
                    -D.Independent(D.Normal(predicted_obs, obs_std), obs.ndim - 2)
                    .log_prob(obs)
                    .mean()
                )
            else:
                accuracy = getattr(LossFunctions, accuracy_metric)(
                    predicted_obs, obs, sum_dim=[-1, -2, -3]
                )
            loss_dict["loss"] = loss_dict["loss"] + (accuracy * alpha)
            loss_dict["accuracy"] = accuracy.item()

        if alpha < 1:
            # latent_states shape: (batch_t * batch_b, latent_dim)
            hidden_feature: torch.Tensor = proj_activation(
                latent_proj(latent_states))
            # obs_embed shape: (batch_t * batch_b, embed_dim)
            obs_feature: torch.Tensor = proj_activation(obs_proj(obs_embed))

            if hidden_feature.ndim == 3:
                hidden_feature = hidden_feature.flatten(0, 1)
            if obs_feature.ndim == 3:
                obs_feature = obs_feature.flatten(0, 1)

            if use_ce:

                labels = (
                    torch.Tensor(list(range(len(hidden_feature))))
                    .long()
                    .to(hidden_feature.device)
                )


                sim_matrix = torch.mm(hidden_feature, obs_feature.T)

                nce_loss = F.cross_entropy(sim_matrix, labels, reduction="mean") - np.log(
                    len(sim_matrix)
                )

            else:
                positive = torch.sum(hidden_feature * obs_feature, dim=-1)
                loss_dict["positive"] = positive.detach().clone().mean().item()

                sim_matrix = torch.mm(hidden_feature, obs_feature.T)

                negative = torch.logsumexp(
                    sim_matrix, dim=-1) - np.log(len(sim_matrix))
                loss_dict["negative"] = negative.detach().clone().mean().item()
                nce_loss = torch.mean(-positive + negative)
            loss_dict["loss"] = loss_dict["loss"] + (nce_loss * (1 - alpha))
            loss_dict["nce"] = nce_loss.item()

        if kl_balancing:
            complexity = LossFunctions.kl_balancing(
                posterior, prior, kl_balancing)
        else:
            complexity = LossFunctions.kl_vanilla(posterior, prior)
        complexity = torch.max(
            complexity, complexity.new_full(complexity.size(), free_nats)
        )
        loss_dict["loss"] = loss_dict["loss"] + (complexity * kl_beta)
        loss_dict["complexity"] = complexity.item()

        return loss_dict

    @staticmethod
    def likelihood_expected(
        predicted_obs: torch.Tensor,
        prior: StochState,
        posterior: StochState,
        preferred_obs: torch.Tensor,
        prior_2: StochState,
        posterior_2: StochState,
        pref_dist: D.Distribution,
        pref_std: float = 0.1,
        no_epistemic: bool = False,
        no_extrinsic: bool = False,
        use_logprob: bool = False,
        pixel: bool = False
    ):
        # print("pref_std", pref_std)
        if use_logprob:
            preferred_obs_dist = D.Independent(
                pref_dist(preferred_obs, pref_std), predicted_obs.ndim - 2
            )
            
            logprob_preferences = preferred_obs_dist.log_prob(predicted_obs)
        else:
            if predicted_obs.ndim == 5:
                mse = F.mse_loss(predicted_obs, preferred_obs.unsqueeze(0).expand_as(predicted_obs), reduction="none").sum([-1, -2, -3])
                logprob_preferences = -mse / (pref_std**2)
                if pixel:
                    # print(predicted_obs.shape)
                    num_samples, l, c, h, w = predicted_obs.shape
                    logprob_preferences = logprob_preferences / (c*h*w)
            else:
                mse = F.mse_loss(predicted_obs, preferred_obs.unsqueeze(0).expand_as(predicted_obs), reduction="none").sum([-1])
                logprob_preferences = -mse / (pref_std**2) / (predicted_obs.shape[0]) / (predicted_obs.shape[-1])
        
        loss = torch.zeros_like(logprob_preferences,
                                device=logprob_preferences.device)

        if no_extrinsic:
            logprob_preferences = logprob_preferences.detach()
        else:
            loss -= logprob_preferences

        prior_dist = get_dist(prior)
        post_dist = get_dist(posterior.detach())
        epistemic_term_1 = D.kl_divergence(post_dist, prior_dist)

        if prior_2 is not None:
            prior_dist_2 = get_dist(prior_2)
            post_dist_2 = get_dist(posterior_2.detach())
            epistemic_term_2 = D.kl_divergence(post_dist_2, prior_dist_2) * 2
            # print("epi_1:", epistemic_term_1)
            # print("epi_2:", epistemic_term_2)
        else:
            epistemic_term_2 = 0
            
        epistemic_term = epistemic_term_1 #+ epistemic_term_2

        # if  predicted_obs.ndim == 5:
        #     epistemic_term = epistemic_term.reshape(
        #         *logprob_preferences.shape
        #     )
        # else:
        #     # epistemic_term = epistemic_term.reshape(10, 10, logprob_preferences.shape[-1]).mean(dim=0)
        # print("ext", logprob_preferences)
        # print(epistemic_term.mean())
        # print(logprob_preferences.shape)
        # print(epistemic_term.shape)
        epistemic_term = epistemic_term.reshape(
            *logprob_preferences.shape
        )

        if no_epistemic:
            epistemic_term = epistemic_term.detach()
        else:
            loss -= epistemic_term
        return dict(
            loss=loss,
            extrinsic=logprob_preferences,
            epistemic=epistemic_term,
            epistemic_1 = epistemic_term_1,
            epistemic_2 = epistemic_term_2
        )

    @staticmethod
    def contrastive_expected(
        obs_embed: torch.Tensor,
        pref_embed: torch.Tensor,
        latent_states: torch.Tensor,
        latent_proj: nn.Module,
        obs_proj: nn.Module,
        proj_activation: Activation,
        nce_balancing: float = 0.5,
        no_nce: bool = False
    ):
        batch_t, batch_b, dim = latent_states.shape
        hidden_feature = proj_activation(latent_proj(latent_states))
        pref_feature = proj_activation(obs_proj(pref_embed)).reshape(
            [batch_t, batch_b, hidden_feature.shape[-1]]
        )

        obs_feature = proj_activation(obs_proj(obs_embed)).reshape(-1,
                                                                   pref_feature.shape[-1])

        pos_loss = torch.sum(hidden_feature * pref_feature, dim=-1).reshape(
            batch_t, batch_b
        )

        obs_feature = torch.cat(
            [pref_feature.reshape([batch_t, batch_b, -1])
             [0], obs_feature], dim=0
        )
        sim_matrix = torch.mm(
            hidden_feature.reshape([batch_t * batch_b, -1]), obs_feature.T
        )

        neg_loss = torch.logsumexp(sim_matrix, dim=1).reshape(
            batch_t, batch_b
        ) - np.log(obs_feature.shape[0])
        nce_loss = -pos_loss + neg_loss

        loss = torch.zeros_like(nce_loss, device=nce_loss.device)
        if no_nce:
            nce_loss = nce_loss.detach()
        else:
            loss = nce_loss

        return dict(
            loss=loss,
            nce=nce_loss,
            pos_nce=pos_loss,
            neg_nce=neg_loss,
        )

    @staticmethod
    def expected_alpha_sub(
    #     latent_states: torch.Tensor,
    #     latent_proj: nn.Module,
    #     obs_proj: nn.Module,
    #     proj_activation: Activation,
        prior: StochState,
        posterior: StochState = None,
        predicted_obs: torch.Tensor = None,
        preferred_obs: torch.Tensor = None,
        prior_2: StochState = None,
        posterior_2: StochState = None,
        pref_dist: D.Distribution = D.Normal,
        pref_std: float = 0.1, #選好精度の逆数
        alpha: float = 1.0,
        pixel: bool=False
        # nce_balancing: float = 0.0,
        # obs_embed: torch.Tensor = None,
        # pref_embed: torch.Tensor = None,
    ):

        contrastive_term = dict(loss=0)
        likelihood_term = dict(loss=0)
        loss_dict = dict()
        
        # if alpha < 1:
        #     contrastive_term = CalcFreeEnergy.contrastive_expected(
        #         obs_embed, pref_embed, latent_states, latent_proj, obs_proj, proj_activation, nce_balancing 
        #     )

        #     loss_dict.update(contrastive_term)
            
        ### ↓ こ れ ↓ ###
        if alpha > 0: # num_samples, length次元持ってる
            likelihood_term = CalcFreeEnergy.likelihood_expected(
                predicted_obs, prior, posterior, preferred_obs, prior_2, posterior_2, pref_dist, pref_std, pixel=pixel
            )
            loss_dict.update(likelihood_term)
        loss_dict["loss"] = (contrastive_term["loss"] * (1 - alpha)) + (
            likelihood_term["loss"] * alpha
        )
        
        #num_samplesもlengthも全部平均
        loss_dict["efe"] = loss_dict["loss"].mean().item()
        # loss_dict["epistemic"] = loss_dict["epistemic"].mean().item()
        # loss_dict["extrinsic"] = loss_dict["extrinsic"].mean().item()
        
        if alpha ==1:
            loss_dict["nce"] = torch.zeros_like(loss_dict["loss"])
        if alpha == 0:
            loss_dict["extrinsic"] = torch.zeros_like(loss_dict["loss"])
            loss_dict["epistemic"] = torch.zeros_like(loss_dict["loss"])
        
        # print(loss_dict["efe"])
        # print(loss_dict["epistemic"])
        # print(loss_dict["extrinsic"])
        return loss_dict

    @staticmethod
    def gae_estimation(
        free_energy, value_preds, gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        lambda_returns = torch.zeros_like(free_energy)
        gamma = torch.ones_like(free_energy) * gamma
        lambda_returns[-1] = free_energy[-1] + gamma[-1] * value_preds[-1]
        for step in reversed(range(free_energy[:-1].size(0))):
            lambda_returns[step] = free_energy[step] + gamma[step] * (
                (1 - gae_lambda) * value_preds[step + 1]
                + gae_lambda * lambda_returns[step + 1]
            )
        return lambda_returns
