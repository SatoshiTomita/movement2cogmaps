import torch
from architectures.rnn_core import RNNModule


class RNN(torch.nn.Module):
    """Encoder-decoder RNN with grid-cell initialised hidden state.

    The first hidden state is derived from grid-cell activations via a linear
    projection, after which the RNN processes scene inputs normally.

    Args:
        device: Torch device for tensor allocation.
        scene_dim: Dimensionality of each scene input timestep.
        gridcells_dim: Dimensionality of the grid-cell input.
        output_dim: Dimensionality of decoder output.
        latent_dim: Hidden state size (default: 500).
        nonlinearity: Activation function name for RNNModule (default: 'sigmoid').
        dropouts: List of dropout rates [input, hidden, decoder] (default: [0,0,0]).
        bias: Whether to use bias in all linear layers (default: False).
    """

    def __init__(self, device, scene_dim, gridcells_dim, output_dim, latent_dim=500,
                 nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False):
        super().__init__()

        self.gc2hidden = torch.nn.Linear(gridcells_dim, latent_dim)
        self.rnn = RNNModule(
            device, scene_dim, latent_dim,
            nonlinearity=nonlinearity, dropouts=dropouts,
            input_bias=bias, hidden_bias=bias
        )

        self.decoder_lin = torch.nn.Linear(latent_dim, output_dim, bias=bias)
        self.add_do = dropouts[-1] > 0
        self.decoder_do = torch.nn.Dropout(dropouts[-1])

    def encode(self, scene, gc, hidden):
        """Encode scene sequence to hidden states, using grid cells for initial state.

        Args:
            scene: Input tensor [batch, time, scene_dim].
            gc: Grid-cell activations [batch, time, gridcells_dim].
            hidden: Previous final hidden state [batch, latent_dim] or None.

        Returns:
            All hidden states [batch, time, latent_dim].
        """
        if hidden is not None:
            return self.rnn(scene, hidden[None, ...])[0]
        # Initialise hidden state from the first grid-cell timestep
        return self.rnn(scene, self.gc2hidden(gc[:, 0, ...]))[0]

    def decode(self, x):
        """Decode hidden states to output space.

        Args:
            x: Hidden states [batch, time, latent_dim].

        Returns:
            Decoded output [batch, time, output_dim].
        """
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, scene, gc, hidden=None):
        """Full forward pass: encode then decode.

        Args:
            scene: Input tensor [batch, time, scene_dim].
            gc: Grid-cell activations [batch, time, gridcells_dim].
            hidden: Optional previous hidden state [batch, latent_dim].

        Returns:
            Tuple of (output [batch, time, output_dim],
                       all hidden states [batch, time, latent_dim],
                       last hidden state [batch, latent_dim]).
        """
        hidden_all = self.encode(scene, gc, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, hidden_all[:, -1, :]