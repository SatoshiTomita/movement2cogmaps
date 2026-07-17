import torch


class GRU(torch.nn.Module):
    """Encoder-decoder GRU for sequence prediction using BPTT.

    Mirrors the interface of ``architectures.recurrent.rnn_bptt.RNN`` so it can
    be used as a drop-in replacement in the training/activity pipeline. Encodes
    input sequences with a ``torch.nn.GRU`` and decodes the hidden states
    through a linear layer with optional dropout.

    Args:
        device: Torch device for tensor allocation.
        input_dim: Dimensionality of each input timestep.
        output_dim: Dimensionality of decoder output.
        latent_dim: Hidden state size (default: 500).
        nonlinearity: Unused, kept for signature compatibility with RNN.
        dropouts: List of dropout rates [input, hidden, decoder] (default: [0,0,0]).
        bias: Whether to use bias in all layers (default: False).
    """

    def __init__(self, device, input_dim, output_dim, latent_dim=500,
                 nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False):
        super().__init__()

        self.device = device
        self.gru = torch.nn.GRU(
            input_dim, latent_dim,
            batch_first=True, bias=bool(bias)
        )

        self.decoder_lin = torch.nn.Linear(latent_dim, output_dim, bias=bool(bias))
        self.add_do = dropouts[-1] > 0
        self.decoder_do = torch.nn.Dropout(dropouts[-1])

    def encode(self, x, hidden):
        """Encode input sequence to hidden states.

        Args:
            x: Input tensor [batch, time, input_dim].
            hidden: Previous final hidden state [batch, latent_dim] or None.

        Returns:
            All hidden states [batch, time, latent_dim].
        """
        if hidden is not None:
            return self.gru(x, hidden[None, ...].contiguous())[0]
        return self.gru(x)[0]

    def decode(self, x):
        """Decode hidden states to output space.

        Args:
            x: Hidden states [batch, time, latent_dim].

        Returns:
            Decoded output [batch, time, output_dim].
        """
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, inputs, hidden=None):
        """Full forward pass: encode then decode.

        Args:
            inputs: Input tensor [batch, time, input_dim].
            hidden: Optional previous hidden state [batch, latent_dim].

        Returns:
            Tuple of (output [batch, time, output_dim],
                       all hidden states [batch, time, latent_dim],
                       last hidden state [batch, latent_dim]).
        """
        hidden_all = self.encode(inputs, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, hidden_all[:, -1, :]
