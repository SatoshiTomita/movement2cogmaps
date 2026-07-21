import torch


class GRU(torch.nn.Module):
    """
    device, input_dim, output_dim, latent_dim=500,nonlinearity='sigmoid', dropouts=[0, 0, 0], bias=False
    [batch,time,input_dim]->[batch,time,latent_dim]->[batch,time,output_dim]
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
        """
        [batch, time, input_dim] -> [batch, time, latent_dim]
        """
        if hidden is not None:
            return self.gru(x, hidden[None, ...].contiguous())[0]
        return self.gru(x)[0]

    def decode(self, x):
        """
        [batch, time, latent_dim] -> [batch, time, output_dim]
        """
        out = self.decoder_lin(x)
        return self.decoder_do(out) if self.add_do else out

    def forward(self, inputs, hidden=None):
        """
        [batch, time, input_dim] -> [batch, time, output_dim]
        """
        hidden_all = self.encode(inputs, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, hidden_all[:, -1, :]
