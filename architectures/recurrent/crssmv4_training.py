"""Train networks.dynamics.CRSSMV4 through CoarseWorldModel."""

import torch

from architectures.losses_custom import get_hidden_l2norm, get_weights_l2norm
from architectures.recurrent.training import TrainerBPTT
from architectures.world import CoarseWorldModel
from utils.config import (
    CNNConfig,
    CRSSMV4Config,
    DistributionConfig,
    GateL0RDConfig,
    RNNConfig,
    WorldConfig,
)


def _distribution(args, stoch_dim, hidden_dim):
    return DistributionConfig(
        stoch_dim=stoch_dim,
        hidden_dim=hidden_dim,
        dist=args.stoch_dist,
        layers=1,
        activation="Mish",
        n_class=args.stoch_n_class if args.stoch_dist == "categorical" else 1,
    )


def build_crssmv4_world_model(args, action_dim, frame_shape):
    """Return an actual CoarseWorldModel containing CRSSMV4."""
    height, width = frame_shape
    dynamics_cfg = CRSSMV4Config(
        determ_dim=args.latent_dim,
        coarse_dim=args.coarse_dim,
        stoch_cfg=_distribution(args, args.stoch_dim, args.latent_dim),
        coarse_stoch_cfg=_distribution(
            args, args.coarse_stoch_dim, args.coarse_dim
        ),
        init_from_="obs",
        init_with_="posterior",
        precise_rnn="GRU",
        coarse_rnn="GateL0RD",
        precise_cfg=RNNConfig(bias=bool(args.bias)),
        coarse_cfg=GateL0RDConfig(
            activation="Mish",
            dense_hidden_dim=args.coarse_hidden_dim,
        ),
        coarse_obs="obs",
    )
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
            "recon_coarse": True,
            "w_coarse_obs": 0.0,
            "w_l0_norm": args.w_l0_norm,
        },
        dynamics_cfg=dynamics_cfg,
        encoder_cfg=encoder_cfg,
        decoder_cfg=decoder_cfg,
        shared_decoder=False,
    )
    return CoarseWorldModel(cfg)


class TrainerCRSSMV4(TrainerBPTT):
    """BPTT loop that operates directly on CoarseWorldModel."""

    def __init__(self, args, optimizer, loss_fn, device):
        super().__init__(args, optimizer, loss_fn, device)
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
                f"CRSSMV4 expected {expected} image features, got {scene.shape[-1]}"
            )
        image = scene.reshape(*scene.shape[:-1], 1, self.height, self.width)
        target_image = target.reshape(
            *target.shape[:-1], 1, self.height, self.width
        )
        return scene, image, action, target, target_image, pos, theta

    def _run_window(self, model, image, action, target_image, initialize):
        batch = (
            action.transpose(0, 1),
            image.transpose(0, 1),
            target_image.transpose(0, 1),
        )
        loss_dict, embed, prediction = model._train(batch, init=initialize)
        loss_dict, prediction = model._coarse_train(
            loss_dict, embed, prediction, batch[-1]
        )
        output = prediction["prediction"].transpose(0, 1).flatten(-3)
        latent = prediction["latent_states"].transpose(0, 1)
        return loss_dict, output, latent

    def _predict_plot_batch(self, model, data, hidden_last):
        scene, image, action, target, target_image, _, _ = self._prepare_batch(data)
        _, output, latent = self._run_window(
            model, image, action, target_image, hidden_last is None
        )
        model.dynamics.detach()
        return torch.cat([scene, action], -1), target.cpu(), output, latent[:, -1]

    def train_epoch(self, model, dataloader):
        model.train()
        totals, initialized = {}, False
        for index, data in enumerate(dataloader):
            self.optimizer.zero_grad()
            _, image, action, _, target_image, _, _ = self._prepare_batch(data)
            loss_dict, _, latent = self._run_window(
                model, image, action, target_image, not initialized
            )
            loss = loss_dict["loss"]
            hidden_l2 = get_hidden_l2norm(latent)
            weights_l2 = get_weights_l2norm(model)
            total = (
                loss
                + self.args.hidden_reg * hidden_l2
                + self.args.weights_reg * weights_l2
            )
            total.backward()
            if self.args.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_value)
            self.optimizer.step()
            totals = self._update_losses(
                ["loss_train", "tot_loss_train"], [loss, total], totals
            )
            for key, value in loss_dict.items():
                if key != "loss":
                    totals = self._update_losses([f"{key}_train"], [value], totals)
            totals = self._update_hidden_layer(
                ["hidden_l2norm"], [hidden_l2], totals
            )
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
                loss = loss_dict["loss"]
                input_loss = self.loss_fn(output, scene)
                distance = self.loss_fn(target, scene)
                totals = self._update_losses(
                    ["loss_test", "loss_wrt_input", "distance_input", "tot_loss_test"],
                    [loss, input_loss, distance, loss], totals,
                )
                for key, value in loss_dict.items():
                    if key != "loss":
                        totals = self._update_losses([f"{key}_test"], [value], totals)
                loss_list.append(float(loss))
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
