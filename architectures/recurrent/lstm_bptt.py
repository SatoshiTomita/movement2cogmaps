import torch


class LSTM(torch.nn.Module):
    """Encoder-decoder LSTM for sequence prediction using BPTT.

    The public interface matches the existing RNN and GRU models.  The returned
    recurrent state is an ``(hidden, cell)`` tuple, allowing the BPTT trainer to
    preserve both components between consecutive windows.
    """

    def __init__(self, device, input_dim, output_dim, latent_dim=500,
                 nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False):
        super().__init__()

        self.device = device
        self.lstm = torch.nn.LSTM(
            input_dim, latent_dim,
            batch_first=True, bias=bool(bias)
        )

        self.decoder_lin = torch.nn.Linear(latent_dim, output_dim, bias=bool(bias))
        self.add_do = dropouts[-1] > 0
        self.decoder_do = torch.nn.Dropout(dropouts[-1])

    def encode(self, x, state=None):
        """Encode a sequence and return all hidden states and final state."""
        if state is not None:
            hidden, cell = state
            return self.lstm(
                x,
                (hidden[None, ...].contiguous(), cell[None, ...].contiguous())
            )
        return self.lstm(x)

    def decode(self, x):
        """Decode hidden states to the output space."""
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, inputs, hidden=None):
        """Run the LSTM and return output, hidden sequence, and final state."""
        hidden_all, (hidden_last, cell_last) = self.encode(inputs, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, (hidden_last[0], cell_last[0])
