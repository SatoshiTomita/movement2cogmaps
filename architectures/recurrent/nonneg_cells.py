"""Single-layer GRU/LSTM whose recurrent states stay non-negative.

They follow the ``torch.nn.GRU`` / ``torch.nn.LSTM`` equations (batch_first,
one layer, same call signature and return values) except that the candidate
nonlinearity is a sigmoid instead of tanh:

* GRU:  n_t = sigmoid(W_in x + r * (W_hn h)), so h_t = (1 - z) n_t + z h_{t-1}
  stays in [0, 1] when h_0 does.
* LSTM: g_t = sigmoid(W_ig x + W_hg h), so c_t = f c_{t-1} + i g_t >= 0 when
  c_0 >= 0, and h_t = o tanh(c_t) lies in [0, 1).

The zero initial state used without grid cells satisfies these conditions.
"""

import math

import torch


class _NonnegRecurrent(torch.nn.Module):
    """Shared parameters in the torch.nn.GRU/LSTM layout (gates stacked)."""

    n_gates = 1

    def __init__(self, input_size, hidden_size, batch_first=True, bias=True):
        super().__init__()
        if not batch_first:
            raise ValueError("Only batch_first=True is supported")
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = self.n_gates * hidden_size
        self.weight_ih_l0 = torch.nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh_l0 = torch.nn.Parameter(torch.empty(gate_size, hidden_size))
        if bias:
            self.bias_ih_l0 = torch.nn.Parameter(torch.empty(gate_size))
            self.bias_hh_l0 = torch.nn.Parameter(torch.empty(gate_size))
        else:
            self.register_parameter("bias_ih_l0", None)
            self.register_parameter("bias_hh_l0", None)
        bound = 1.0 / math.sqrt(hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -bound, bound)

    def _input_projection(self, x):
        return torch.nn.functional.linear(x, self.weight_ih_l0, self.bias_ih_l0)

    def _hidden_projection(self, h):
        return torch.nn.functional.linear(h, self.weight_hh_l0, self.bias_hh_l0)


class NonnegGRU(_NonnegRecurrent):
    n_gates = 3

    def forward(self, x, h0=None):
        """[batch, time, input] -> ([batch, time, hidden], [1, batch, hidden])."""
        if h0 is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
        else:
            h = h0[0]
        gi_all = self._input_projection(x)
        outputs = []
        for t in range(x.shape[1]):
            gi_r, gi_z, gi_n = gi_all[:, t].chunk(3, dim=-1)
            gh_r, gh_z, gh_n = self._hidden_projection(h).chunk(3, dim=-1)
            r = torch.sigmoid(gi_r + gh_r)
            z = torch.sigmoid(gi_z + gh_z)
            n = torch.sigmoid(gi_n + r * gh_n)
            h = (1 - z) * n + z * h
            outputs.append(h)
        return torch.stack(outputs, dim=1), h[None]


class NonnegLSTM(_NonnegRecurrent):
    n_gates = 4

    def forward(self, x, state=None):
        """[batch, time, input] -> ([batch, time, hidden], (h_n, c_n))."""
        if state is None:
            h = x.new_zeros(x.shape[0], self.hidden_size)
            c = x.new_zeros(x.shape[0], self.hidden_size)
        else:
            h, c = state[0][0], state[1][0]
        gi_all = self._input_projection(x)
        outputs = []
        for t in range(x.shape[1]):
            gates = gi_all[:, t] + self._hidden_projection(h)
            i, f, g, o = gates.chunk(4, dim=-1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.sigmoid(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h[None], c[None])
