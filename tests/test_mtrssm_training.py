import unittest
from types import SimpleNamespace

import torch
from torch import nn

from architectures.recurrent.mtrssm_training import MTRSSMPredictor
from architectures.recurrent.training import TrainerMTRSSM
from utils.config import (
    DistributionConfig,
    MTRNNConfig,
    MTRSSMConfig,
    RSSMConfig,
)


def _rssm_level(determ_dim, stoch_dim, tau):
    return RSSMConfig(
        determ_dim=determ_dim,
        stoch_cfg=DistributionConfig(
            stoch_dim=stoch_dim,
            hidden_dim=4,
            dist="normal",
            layers=1,
            activation="Mish",
        ),
        init_from_="obs",
        init_with_="posterior",
        rnn_name="MTRNN",
        rnn_cfg=MTRNNConfig(tau=tau),
    )


def _make_model(identity_decoder=True):
    cfg = MTRSSMConfig(
        lower_cfg=_rssm_level(2, 2, 2),
        higher_cfg=_rssm_level(2, 2, 4),
        temporal_abstraction=2,
        top_obs="determ",
    )
    model = MTRSSMPredictor(obs_dim=8, action_dim=2, cfg=cfg)
    if identity_decoder:
        model.decoder = nn.Identity()
    return model


class _RecordingL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.targets = []

    def forward(self, outputs, targets):
        self.targets.append(targets.detach().clone())
        return torch.mean(torch.abs(outputs - targets))


class MTRSSMTrainingTest(unittest.TestCase):
    def test_current_time_reconstruction_and_two_level_state_carry(self):
        torch.manual_seed(3)
        model = _make_model()
        batch_size, window_size = 2, 5

        outputs, hidden_all, hidden_last = model.observe(
            action=torch.randn(batch_size, window_size, 2),
            observation=torch.randn(batch_size, window_size, 8),
            initial_obs=torch.randn(batch_size, 8),
        )

        self.assertEqual(outputs.shape, (batch_size, window_size, 8))
        self.assertTrue(torch.equal(outputs, hidden_all))
        self.assertEqual(hidden_last.shape, (batch_size, 8))
        self.assertEqual(model.last_prior.stoch.shape[0], window_size)
        self.assertEqual(model.last_high_prior.stoch.shape[0], 3)

        next_outputs, _, _ = model.observe(
            action=torch.randn(batch_size, 2, 2),
            observation=torch.randn(batch_size, 2, 8),
            state=hidden_last,
        )
        self.assertTrue(torch.allclose(next_outputs[:, 0], hidden_last))

    def test_training_uses_scene_and_both_kl_levels(self):
        torch.manual_seed(4)
        model = _make_model(identity_decoder=False)
        loss_fn = _RecordingL1Loss()
        trainer = TrainerMTRSSM(
            SimpleNamespace(
                free_nats=0.0,
                kl_scale=1.0,
                high_kl_scale=0.5,
                clip_value=None,
                reset_hidden_at=None,
            ),
            torch.optim.SGD(model.parameters(), lr=1e-3),
            loss_fn,
            torch.device("cpu"),
        )

        batch_size, window_size = 2, 3
        scene = torch.randn(1, batch_size, window_size, 8)
        velocity = torch.randn(1, batch_size, 1, window_size, 1)
        rotational_velocity = torch.randn(1, batch_size, 1, window_size, 1)
        positions = torch.randn(1, batch_size, 1, window_size, 2)
        thetas = torch.randn(1, batch_size, 1, window_size, 1)
        labels = torch.randn(1, batch_size, 1, window_size, 8)

        window = (
            scene,
            velocity,
            rotational_velocity,
            positions,
            thetas,
            labels,
        )
        # Two windows exercise truncated-BPTT state restoration for both MTRNN
        # levels; a retained graph would fail on the second backward pass.
        trainer.train_epoch(model, [window, window])

        self.assertTrue(torch.equal(loss_fn.targets[0], scene.squeeze(0)))
        self.assertTrue(torch.equal(loss_fn.targets[1], scene.squeeze(0)))
        self.assertTrue(hasattr(model, "last_low_kl"))
        self.assertTrue(hasattr(model, "last_high_kl"))


if __name__ == "__main__":
    unittest.main()
