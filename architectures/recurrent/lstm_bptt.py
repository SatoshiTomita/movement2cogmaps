import torch
import torch.nn.functional as F


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

    def unfold_states(self, inputs, state=None):
        """Return the per-timestep ``(h_t, c_t)`` sequences.

        ``torch.nn.LSTM`` exposes every ``h_t`` but only the final ``c_t``.
        Re-evaluate the documented single-layer LSTM equations with the same
        parameters so activity analysis can inspect the complete cell-state
        trajectory without changing the trained model.
        """
        batch_size = inputs.shape[0]
        hidden_size = self.lstm.hidden_size
        if state is None:
            hidden = inputs.new_zeros((batch_size, hidden_size))
            cell = inputs.new_zeros((batch_size, hidden_size))
        else:
            hidden, cell = state

        hidden_sequence = []
        cell_sequence = []
        for timestep in range(inputs.shape[1]):
            gates = F.linear(
                inputs[:, timestep],
                self.lstm.weight_ih_l0,
                getattr(self.lstm, "bias_ih_l0", None),
            ) + F.linear(
                hidden,
                self.lstm.weight_hh_l0,
                getattr(self.lstm, "bias_hh_l0", None),
            )
            input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
            input_gate = torch.sigmoid(input_gate)
            forget_gate = torch.sigmoid(forget_gate)
            candidate = torch.tanh(candidate)
            output_gate = torch.sigmoid(output_gate)
            cell = forget_gate * cell + input_gate * candidate
            hidden = output_gate * torch.tanh(cell)
            hidden_sequence.append(hidden)
            cell_sequence.append(cell)

        return (
            torch.stack(hidden_sequence, dim=1),
            torch.stack(cell_sequence, dim=1),
        )

    def forward(self, inputs, hidden=None):
        """Run the LSTM and return output, hidden sequence, and final state."""
        hidden_all, (hidden_last, cell_last) = self.encode(inputs, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, (hidden_last[0], cell_last[0])
