"""Train RSSM and MTRSSM directly through architectures.world.WorldModel."""

import torch

from architectures.losses_custom import get_hidden_l2norm, get_weights_l2norm
from architectures.recurrent.training import TrainerBPTT
from architectures.world import WorldModel
from networks.dynamics import MTRSSM
from utils.config import (
    CNNConfig,
    DistributionConfig,
    MTRNNConfig,
    MTRSSMConfig,
    RNNConfig,
    RSSMConfig,
    WorldConfig,
)
from utils.loss import LossFunctions


def _distribution(args, stoch_dim, hidden_dim):
    return DistributionConfig(
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
        dist=args.stoch_dist,
        layers=1,
        activation="Mish",
        n_class=args.stoch_n_class if args.stoch_dist == "categorical" else 1,
    )


def _rssm_config(args, determ_dim, stoch_dim, rnn_name, tau=None):
    if rnn_name == "MTRNN":
        rnn_cfg = MTRNNConfig(tau=tau, bias=bool(args.bias))
    else:
        rnn_cfg = RNNConfig(bias=bool(args.bias))
    return RSSMConfig(
        determ_dim=determ_dim,
        stoch_cfg=_distribution(args, stoch_dim, determ_dim),
        init_from_="obs",
        init_with_="posterior",
        rnn_name=rnn_name,
        rnn_cfg=rnn_cfg,
    )


def build_world_model(args, architecture, action_dim, frame_shape):
    """Build a WorldModel containing the real RSSM or MTRSSM dynamics."""
    if architecture == "rssm":
        dynamics_cfg = _rssm_config(
            args, args.latent_dim, args.stoch_dim, "GRU"
        )
    elif architecture == "mtrssm":
        dynamics_cfg = MTRSSMConfig(
            lower_cfg=_rssm_config(
                args,
                args.latent_dim,
                args.stoch_dim,
                "MTRNN",
                args.lower_tau,
            ),
            higher_cfg=_rssm_config(
                args,
                args.higher_latent_dim,
                args.higher_stoch_dim,
                "MTRNN",
                args.higher_tau,
            ),
            temporal_abstraction=args.temporal_abstraction,
            top_obs=args.top_obs,
        )
    else:
        raise ValueError(f"Unsupported WorldModel architecture: {architecture}")

    height, width = frame_shape
    encoder_cfg = CNNConfig(
        channels=(8, 16),
        kernels=(4, 4),
        strides=(2, 2),
        paddings=(1, 1),
        hidden_activation="Mish",
        output_activation="Tanh",
    )
    decoder_cfg = CNNConfig(
        channels=(16, 8),
        kernels=(4, 4),
        strides=(2, 2),
        paddings=(1, 1),
        hidden_activation="Mish",
        output_activation="Sigmoid",
    )
    cfg = WorldConfig(
        obs_shape=(1, height, width),
        obs_dim=args.embed_obs_dim,
        action_dim=action_dim,
        alpha=1.0,
        activation="Mish",
        optimizer_cfg={"name": "RMSprop", "lr": args.lr},
        loss_cfg={
            "embed_beta": 0.0,
            "kl_balancing": 0.0,
            "kl_beta": args.kl_scale,
            "free_nats": args.free_nats,
            "obs_std": args.obs_std,
            "accuracy_metric": "logprob",
        },
        dynamics_cfg=dynamics_cfg,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
    )
    return WorldModel(cfg)


class TrainerWorldModel(TrainerBPTT):
    """BPTT loop for a real WorldModel, without a Predictor adapter."""

    def __init__(self, args, optimizer, loss_fn, device):
        super().__init__(args, optimizer, loss_fn, device)
        if args.n_future_pred != 1:
            raise ValueError(
                "RSSM/MTRSSM WorldModel training currently requires "
                "n_future_pred=1"
            )
        self.height = args.frame_dim[1] // args.frame_subsampling
        self.width = args.frame_dim[0] // args.frame_subsampling

    def _prepare_batch(self, data):
        scene, velocity, rotational_velocity, pos, theta, labels = data
        scene = scene.squeeze(0).to(self.device)
        target = labels.squeeze(0)[:, 0].to(self.device)
        action = torch.cat(
            [
                velocity.squeeze(0)[:, 0].to(self.device),
                rotational_velocity.squeeze(0)[:, 0].to(self.device),
            ],
            dim=-1,
        )
        expected = self.height * self.width
        if scene.shape[-1] != expected:
            raise ValueError(
                f"WorldModel expected {expected} image features, "
                f"got {scene.shape[-1]}"
            )
        image = scene.reshape(*scene.shape[:-1], 1, self.height, self.width)
        target_image = target.reshape(
            *target.shape[:-1], 1, self.height, self.width
        )
        return scene, image, action, target, target_image, pos, theta

    def _run_window(self, model, image, action, target_image, initialize):
        # a_t,o_(t+1),o_t,o_0
        batch = (
            action.transpose(0, 1),
            target_image.transpose(0, 1),
            image.transpose(0, 1),
            image[:, 0] if initialize else None,
        )
        loss_dict, _, prediction = model._train(batch, init=initialize)
        if isinstance(model.dynamics, MTRSSM):
            worlds = prediction["world_states"]
            top_complexity = LossFunctions.kl_vanilla(
                worlds.layer1.posterior, worlds.layer1.prior
            )
            top_complexity = torch.clamp(
                top_complexity, min=self.args.free_nats
            )
            loss_dict["loss"] = loss_dict["loss"] + (
                self.args.high_kl_scale - 1.0
            ) * top_complexity
            loss_dict["scaled_top_complexity"] = float(
                self.args.high_kl_scale * top_complexity.detach()
            )
        output = prediction["prediction"].transpose(0, 1).flatten(-3)
        # Positions/headings and reconstruction targets both describe the
        # pre-action/current times, so use the same aligned state history.
        latent = prediction["analysis_latent_states"].transpose(0, 1)
        return loss_dict, output, latent

    def _predict_plot_batch(self, model, data, hidden_last):
        scene, image, action, target, target_image, _, _ = self._prepare_batch(data)
        _, output, latent = self._run_window(
            model, image, action, target_image, hidden_last is None
        )
        model.dynamics.detach()
        return torch.cat([scene, action], -1), scene.cpu(), output, latent[:, -1]

    def train_epoch(self, model, dataloader):
        model.train()
        totals, initialized = {}, False
        for index, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            _, image, action, _, target_image, _, _ = self._prepare_batch(data)
            loss_dict, _, latent = self._run_window(
                model, image, action, target_image, not initialized
            )
            model_loss = loss_dict["loss"]
            hidden_l2 = get_hidden_l2norm(latent)
            weights_l2 = get_weights_l2norm(model)
            total = (
                model_loss
                + self.args.hidden_reg * hidden_l2
                + self.args.weights_reg * weights_l2
            )
            total.backward()
            if self.args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_value)
            self.optimizer.step()
            totals = self._update_losses(
                ["loss_train", "tot_loss_train"], [model_loss, total], totals
            )
            for key, value in loss_dict.items():
                if key != "loss":
                    totals = self._update_losses([f"{key}_train"], [value], totals)
            totals = self._update_hidden_layer(["hidden_l2norm"], [hidden_l2], totals)
            totals = self._update_norms(model, totals)
            model.dynamics.detach()
            initialized = True
            if self.args.reset_hidden_at and (index + 1) % self.args.reset_hidden_at == 0:
                initialized = False
        if len(dataloader) == 0:
            raise ValueError("Dataloader is empty.")
        return model, {key: value / len(dataloader) for key, value in totals.items()}

    def test_epoch(self, model, dataloader, for_trajectory=False):
        model.eval()
        totals, initialized = {}, False
        loss_list, input_loss_list, distance_list = [], [], []
        if for_trajectory:
            hidden_activity, positions, thetas = [], [], []
        with torch.no_grad():
            for index, data in enumerate(dataloader):
                scene, image, action, target, target_image, pos, theta = self._prepare_batch(data)
                loss_dict, output, latent = self._run_window(
                    model, image, action, target_image, not initialized
                )
                model_loss = loss_dict["loss"]
                input_loss = self.loss_fn(output, scene)
                distance = self.loss_fn(target, scene)
                totals = self._update_losses(
                    ["loss_test", "loss_wrt_input", "distance_input", "tot_loss_test"],
                    [model_loss, input_loss, distance, model_loss], totals,
                )
                for key, value in loss_dict.items():
                    if key != "loss":
                        totals = self._update_losses([f"{key}_test"], [value], totals)
                loss_list.append(float(model_loss))
                input_loss_list.append(float(input_loss))
                distance_list.append(float(distance))
                model.dynamics.detach()
                initialized = True
                if self.args.reset_hidden_at and (index + 1) % self.args.reset_hidden_at == 0:
                    initialized = False
                if for_trajectory:
                    hidden_activity.append(latent.cpu().numpy())
                    positions.append(pos.squeeze(0)[:, 0].cpu().numpy())
                    thetas.append(theta.squeeze(0)[:, 0].cpu().numpy())
        if len(dataloader) == 0:
            raise ValueError("Dataloader is empty.")
        totals = {key: value / len(dataloader) for key, value in totals.items()}
        if for_trajectory:
            return (
                totals, hidden_activity, positions, thetas,
                loss_list, input_loss_list, distance_list,
            )
        return totals
