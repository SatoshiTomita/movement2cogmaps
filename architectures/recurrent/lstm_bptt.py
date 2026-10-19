import torch

from architectures.recurrent.nonneg_cells import NonnegLSTM


class LSTM(torch.nn.Module):
    """Encoder-decoder LSTM for sequence prediction using BPTT.

    The public interface matches the existing RNN and GRU models.  The returned
    recurrent state is an ``(hidden, cell)`` tuple, allowing the BPTT trainer to
    preserve both components between consecutive windows.
    """

    def __init__(self, device, input_dim, output_dim, latent_dim=500,
                 nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False,
                 gridcells_dim=None, nonneg_hidden=False):
        super().__init__()

        self.device = device
        recurrent_layer = NonnegLSTM if nonneg_hidden else torch.nn.LSTM
        self.lstm = recurrent_layer(
            input_dim, latent_dim,
            batch_first=True, bias=bool(bias)
        )
        self.gc2hidden = (
            torch.nn.Linear(gridcells_dim, latent_dim)
            if gridcells_dim is not None else None
        )
        self.gc2cell = (
            torch.nn.Linear(gridcells_dim, latent_dim)
            if gridcells_dim is not None else None
        )

        self.decoder_lin = torch.nn.Linear(latent_dim, output_dim, bias=bool(bias))
        self.add_do = dropouts[-1] > 0
        self.decoder_do = torch.nn.Dropout(dropouts[-1])

    def initial_state(self, x, state=None, gridcells=None):
        """Return the ``(1, batch, latent)`` initial state, or None for zeros."""
        if state is not None:
            hidden, cell = state
            return hidden[None, ...].contiguous(), cell[None, ...].contiguous()
        # Checkpoints pickled before grid-cell support lack gc2hidden.
        if getattr(self, "gc2hidden", None) is not None:
            if gridcells is None:
                raise ValueError("gridcells must be provided when grid-cell input is enabled")
            gc0 = gridcells[:, 0]
            hidden = self.gc2hidden(gc0)[None, ...].contiguous()
            cell = self.gc2cell(gc0)[None, ...].contiguous()
            return hidden, cell
        return None

    def encode(self, x, state=None, gridcells=None):
        """Encode a sequence and return all hidden states and final state."""
        return self.lstm(x, self.initial_state(x, state, gridcells))

    def encode_with_cells(self, x, state=None, gridcells=None):
        """Encode step by step, also returning the cell state at every step.

        ``torch.nn.LSTM`` only exposes the final cell state, so the sequence is
        unrolled one step at a time with the same module and weights.
        """
        state = self.initial_state(x, state, gridcells)
        hidden_steps, cell_steps = [], []
        for t in range(x.shape[1]):
            _, state = self.lstm(x[:, t:t + 1], state)
            hidden_steps.append(state[0][0])
            cell_steps.append(state[1][0])
        hidden_all = torch.stack(hidden_steps, dim=1)
        cell_all = torch.stack(cell_steps, dim=1)
        return hidden_all, cell_all, state

    def decode(self, x):
        """Decode hidden states to the output space."""
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, inputs, hidden=None, gridcells=None):
        """Run the LSTM and return output, hidden sequence, and final state.

        When ``record_cell_state`` is set (activity analysis only), the cell
        sequence of this call is kept in ``cell_all`` for the trainer.
        """
        if getattr(self, "record_cell_state", False):
            hidden_all, self.cell_all, (hidden_last, cell_last) = (
                self.encode_with_cells(inputs, hidden, gridcells)
            )
        else:
            hidden_all, (hidden_last, cell_last) = self.encode(
                inputs, hidden, gridcells
            )
        output = self.decode(hidden_all)
        return output, hidden_all, (hidden_last[0], cell_last[0])
