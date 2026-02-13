import torch
from architectures.activations import (
    MexicanHat, MexicanHatStandard, HardSigmoid, StraightThroughEstimator, HardSoftmax
)

ACTIVATIONS = {
    'sigmoid': lambda n: torch.sigmoid,
    'mexicanhat': lambda n: MexicanHat(),
    'mexicanhatstd': lambda n: MexicanHatStandard(),
    'relu': lambda n: torch.relu,
    'hard_sigmoid': lambda n: HardSigmoid(),
    'step': lambda n: StraightThroughEstimator(),
    'softmax': lambda n: torch.nn.Softmax(dim=1),
    'hard_softmax': lambda n: HardSoftmax(n),
}


class RNNCell(torch.nn.Module):
    """Single-step Elman RNN cell with configurable activation and dropout.

    Args:
        n_inputs: Dimensionality of the input features.
        n_hidden: Dimensionality of the hidden state.
        nonlinearity: Name of the activation function (e.g. 'sigmoid', 'relu').
        dropouts: Tuple/list of dropout rates for (input-to-hidden, hidden-to-hidden).
        input_bias: Whether the input-to-hidden linear layer uses a bias.
        hidden_bias: Whether the hidden-to-hidden linear layer uses a bias.
    """

    def __init__(self, n_inputs, n_hidden, nonlinearity, dropouts, input_bias, hidden_bias):
        super().__init__()

        if nonlinearity not in ACTIVATIONS:
            print("[!!!] WARNING: activation function not recognized, using identity")
        self.activation_fn = ACTIVATIONS.get(nonlinearity, lambda n: torch.nn.Identity())(n_hidden)

        self.in2hidden = torch.nn.Linear(n_inputs, n_hidden, bias=input_bias)
        self.hidden2hidden = torch.nn.Linear(n_hidden, n_hidden, bias=hidden_bias)

        self.add_in2h_do = dropouts[0] > 0
        self.in2h_do = torch.nn.Dropout(dropouts[0])
        self.add_h2h_do = dropouts[1] > 0
        self.h2h_do = torch.nn.Dropout(dropouts[1])

    def forward(self, x, hidden):
        """Compute one recurrent step: h' = activation(W_ih @ x + W_hh @ h).

        Args:
            x: Input tensor [batch, n_inputs].
            hidden: Previous hidden state [batch, n_hidden].

        Returns:
            New hidden state [batch, n_hidden].
        """
        igates = self.in2hidden(x)
        if self.add_in2h_do:
            igates = self.in2h_do(igates)
        hgates = self.hidden2hidden(hidden)
        if self.add_h2h_do:
            hgates = self.h2h_do(hgates)
        return self.activation_fn(igates + hgates)


class RNNModule(torch.nn.Module):
    """Unrolled RNN that applies an RNNCell over the time dimension.

    Args:
        device: Torch device for tensor allocation.
        n_inputs: Dimensionality of input features.
        n_hidden: Dimensionality of the hidden state.
        nonlinearity: Activation function name passed to RNNCell.
        dropouts: Dropout rates passed to RNNCell.
        input_bias: Bias flag for input-to-hidden layer.
        hidden_bias: Bias flag for hidden-to-hidden layer.
    """

    def __init__(self, device, n_inputs, n_hidden, nonlinearity, dropouts,
                 input_bias=True, hidden_bias=True):
        super().__init__()
        self.rnn_cell = RNNCell(n_inputs, n_hidden, nonlinearity, dropouts, input_bias, hidden_bias)
        self.n_hidden = n_hidden
        self.device = device

    def forward(self, x, hidden=None):
        """Run the RNN over the full time dimension.

        Args:
            x: Input tensor [batch, time, n_features].
            hidden: Optional initial hidden state [batch, n_hidden].

        Returns:
            Tuple of (all hidden states [batch, time, n_hidden],
                       last hidden state [batch, n_hidden]).
        """
        batch, time, _ = x.shape
        output = torch.zeros(batch, time, self.n_hidden, device=self.device)
        h = hidden if hidden is not None else torch.zeros(batch, self.n_hidden, device=self.device)

        for t in range(time):
            h = self.rnn_cell(x[:, t, ...], h)
            output[:, t, ...] = h

        return output, h
    