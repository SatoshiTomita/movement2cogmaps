import torch

from architectures.recurrent.nonneg_cells import NonnegGRU


class GRU(torch.nn.Module):
    """
    device, input_dim, output_dim, latent_dim=500,nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False
    [batch,time,input_dim]->[batch,time,latent_dim]->[batch,time,output_dim]
    """

    def __init__(self, device, input_dim, output_dim, latent_dim=500,
                 nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False,
                 gridcells_dim=None, nonneg_hidden=False):
        super().__init__()

        self.device = device
        recurrent_layer = NonnegGRU if nonneg_hidden else torch.nn.GRU
        self.gru = recurrent_layer(
            input_dim, latent_dim,
            batch_first=True, bias=bool(bias)
        )
        self.gc2hidden = (
            torch.nn.Linear(gridcells_dim, latent_dim)
            if gridcells_dim is not None else None
        )

        self.decoder_lin = torch.nn.Linear(latent_dim, output_dim, bias=bool(bias))
        self.add_do = dropouts[-1] > 0
        self.decoder_do = torch.nn.Dropout(dropouts[-1])

    def encode(self, x, hidden, gridcells=None):
        """
        [batch, time, input_dim] -> [batch, time, latent_dim]
        """
        if hidden is not None:
            return self.gru(x, hidden[None, ...].contiguous())[0]
        # Checkpoints pickled before grid-cell support lack gc2hidden.
        if getattr(self, "gc2hidden", None) is not None:
            if gridcells is None:
                raise ValueError("gridcells must be provided when grid-cell input is enabled")
            initial_hidden = self.gc2hidden(gridcells[:, 0])[None, ...]
            return self.gru(x, initial_hidden.contiguous())[0]
        return self.gru(x)[0]

    def decode(self, x):
        """
        [batch, time, latent_dim] -> [batch, time, output_dim]
        """
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, inputs, hidden=None, gridcells=None):
        """
        [batch, time, input_dim] -> [batch, time, output_dim]
        """
        hidden_all = self.encode(inputs, hidden, gridcells)
        output = self.decode(hidden_all)
        return output, hidden_all, hidden_all[:, -1, :]
