import torch


class RSSM(torch.nn.Module):
    """Recurrent State-Space Model (RSSM) for next-step scene prediction.

    Drop-in replacement for ``architectures.recurrent.rnn_bptt.RNN`` in the
    training/activity pipeline. The concatenated ``inputs`` tensor
    ``[scene, velocity, rot_velocity]`` is split internally into the
    observation (scene) and the action (velocities). The recurrent state is
    packed as a single tensor ``concat(determ, stoch)`` so it can be carried
    over across BPTT windows exactly like the plain RNN hidden state, and so
    the downstream place/HD analysis (which reads ``hidden_all``) works
    unchanged with ``latent_dim = determ_dim + stoch_dim``.

    Following the movement2cogmaps convention, the decoder predicts the *next*
    frame from the latent state, so the target stays the next scene frame and
    the existing ``DiscountLoss`` can be reused. The KL term between prior and
    posterior is exposed through ``last_prior`` / ``last_posterior`` attributes
    after each forward pass.

    Args:
        device: Torch device for tensor allocation.
        obs_dim: Dimensionality of the observation (scene) features.
        action_dim: Dimensionality of the action (velocity + rot velocity).
        output_dim: Dimensionality of the decoded output (next scene).
        determ_dim: Deterministic (GRU) hidden state size (default: 500).
        stoch_dim: Stochastic latent size (default: 32).
        hidden_units: Width of the prior/posterior/encoder MLPs (default: 200).
        min_std: Minimum standard deviation for the latent distributions.
        bias: Whether to use bias in the decoder (default: False).
    """

    def __init__(self, device, obs_dim, action_dim, output_dim,
                 determ_dim=500, stoch_dim=32, hidden_units=200,
                 min_std=0.1, bias=False):
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.determ_dim = determ_dim
        self.stoch_dim = stoch_dim
        self.min_std = min_std

        # deterministic transition: h_t = GRUCell([z_{t-1}, a_t], h_{t-1})
        self.cell = torch.nn.GRUCell(stoch_dim + action_dim, determ_dim, bias=True)

        # prior p(z_t | h_t)
        self.prior_net = torch.nn.Sequential(
            torch.nn.Linear(determ_dim, hidden_units),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_units, 2 * stoch_dim),
        )

        # observation encoder enc(o_t)
        self.obs_encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_units),
            torch.nn.ELU(),
        )

        # posterior q(z_t | h_t, enc(o_t))
        self.post_net = torch.nn.Linear(determ_dim + hidden_units, 2 * stoch_dim)

        # decoder: [h_t, z_t] -> next scene
        self.decoder_lin = torch.nn.Linear(determ_dim + stoch_dim, output_dim, bias=bias)

        # populated on every forward pass; read by the RSSM trainer for the KL
        self.last_prior = None
        self.last_posterior = None

    def _dist(self, params):
        """Build a diagonal Gaussian from concatenated (mean, raw_std) params."""
        mean, std = torch.chunk(params, 2, dim=-1)
        std = torch.nn.functional.softplus(std) + self.min_std
        return torch.distributions.Normal(mean, std)

    def _split_inputs(self, inputs):
        """Split the concatenated pipeline input into (obs, action)."""
        obs = inputs[..., :self.obs_dim]
        action = inputs[..., self.obs_dim:]
        return obs, action

    def forward(self, inputs, hidden=None):
        """Run the RSSM over a window and predict the next frame at each step.

        Args:
            inputs: Concatenated tensor [batch, time, obs_dim + action_dim].
            hidden: Optional packed recurrent state [batch, determ_dim + stoch_dim]
                    from the previous window (as returned by this method).

        Returns:
            Tuple of (outputs [batch, time, output_dim],
                       all latent states [batch, time, determ_dim + stoch_dim],
                       last latent state [batch, determ_dim + stoch_dim]).
        """
        obs, action = self._split_inputs(inputs)
        batch, time, _ = obs.shape

        if hidden is not None:
            h = hidden[:, :self.determ_dim]
            z = hidden[:, self.determ_dim:]
        else:
            h = torch.zeros(batch, self.determ_dim, device=self.device)
            z = torch.zeros(batch, self.stoch_dim, device=self.device)

        latents = []
        prior_means, prior_stds = [], []
        post_means, post_stds = [], []

        for t in range(time):
            h = self.cell(torch.cat([z, action[:, t, ...]], dim=-1), h)

            prior = self._dist(self.prior_net(h))
            embed = self.obs_encoder(obs[:, t, ...])
            posterior = self._dist(self.post_net(torch.cat([h, embed], dim=-1)))

            # sample during training, use the mean for deterministic analysis
            z = posterior.rsample() if self.training else posterior.mean

            latents.append(torch.cat([h, z], dim=-1))
            prior_means.append(prior.mean)
            prior_stds.append(prior.stddev)
            post_means.append(posterior.mean)
            post_stds.append(posterior.stddev)

        hidden_all = torch.stack(latents, dim=1)
        outputs = self.decoder_lin(hidden_all)

        self.last_prior = torch.distributions.Normal(
            torch.stack(prior_means, dim=1), torch.stack(prior_stds, dim=1)
        )
        self.last_posterior = torch.distributions.Normal(
            torch.stack(post_means, dim=1), torch.stack(post_stds, dim=1)
        )

        return outputs, hidden_all, hidden_all[:, -1, :]
