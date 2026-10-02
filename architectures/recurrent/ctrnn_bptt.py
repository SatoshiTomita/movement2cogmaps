"""Continuous-time RNN baseline trained with truncated BPTT.

The continuous dynamics are discretised with a forward Euler step.  Time is
measured in data steps, so ``tau`` is also expressed in data steps::

    u_t = (1 - 1 / tau) * u_(t-1)
          + (1 / tau) * (W_in x_t + W_rec h_(t-1) + b)
    h_t = activation(u_t)

Both the membrane potential ``u`` and activity ``h`` are carried between BPTT
windows.  Only ``h`` is decoded and returned as neural activity for the
spatial analyses.
"""

import torch

from architectures.rnn_core import ACTIVATIONS


class CTRNNCell(torch.nn.Module):
    """One Euler-discretised continuous-time recurrent step.
       membraneがニューロンの内部状態u_t、input_dciveがW_in x_t + b_in、recurrent_driveがW_rec h_(t-1) + b_recに対応する
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        tau=4.0,
        nonlinearity="sigmoid",
        dropouts=(0, 0),
        bias=False,
    ):
        super().__init__()

        if tau < 1.0:
            raise ValueError(
                f"CTRNN tau must be at least 1 data step, but got {tau}"
            )
        if nonlinearity not in ACTIVATIONS:
            raise ValueError(
                f"Unknown CTRNN nonlinearity {nonlinearity!r}; "
                f"choose one of {sorted(ACTIVATIONS)}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # activation_fnはニューロンの出力h_tを計算するための活性化関数で、membrane(u_t)に適用される
        self.activation_fn = ACTIVATIONS[nonlinearity](hidden_dim)
        self.in2hidden = torch.nn.Linear(input_dim, hidden_dim, bias=bias)
        self.hidden2hidden = torch.nn.Linear(
            hidden_dim, hidden_dim, bias=bias
        )
        self.in2h_dropout = torch.nn.Dropout(dropouts[0])
        self.h2h_dropout = torch.nn.Dropout(dropouts[1])

        # A buffer follows the model across devices and is saved in its state
        # dict, while remaining fixed during optimisation.
        self.register_buffer("tau", torch.tensor(float(tau)))

    def initial_state(self, inputs):
        """Return zero membrane potential and zero initial activity."""
        shape = (inputs.shape[0], self.hidden_dim)
        membrane = inputs.new_zeros(shape)
        activity = inputs.new_zeros(shape)
        return membrane, activity

    def forward(self, inputs, state=None):
        """Return ``(activity, (membrane, activity))`` for one time step."""
        if state is None:
            membrane, previous_activity = self.initial_state(inputs)
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError(
                    "CTRNN state must be a (membrane, activity) tuple"
                )
            membrane, previous_activity = state

        input_drive = self.in2h_dropout(self.in2hidden(inputs))
        recurrent_drive = self.h2h_dropout(
            self.hidden2hidden(previous_activity)
        )
        alpha = self.tau.reciprocal()
        membrane = (1.0 - alpha) * membrane + alpha * (
            input_drive + recurrent_drive
        )
        activity = self.activation_fn(membrane)
        return activity, (membrane, activity)


class CTRNN(torch.nn.Module):
    """Encoder-decoder CTRNN with the same interface as the RNN baseline."""

    def __init__(
        self,
        device,
        input_dim,
        output_dim,
        latent_dim=500,
        tau=4.0,
        nonlinearity="sigmoid",
        dropouts=(0, 0, 0),
        bias=False,
    ):
        super().__init__()
        if len(dropouts) != 3:
            raise ValueError(
                "CTRNN dropouts must contain [input, recurrent, decoder]"
            )

        # ``device`` is accepted for API compatibility with the other models;
        # state allocation follows the input tensor's actual device.
        self.device = device
        self.latent_dim = latent_dim
        self.ctrnn_cell = CTRNNCell(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            tau=tau,
            nonlinearity=nonlinearity,
            dropouts=dropouts[:2],
            bias=bias,
        )
        self.decoder_lin = torch.nn.Linear(
            latent_dim, output_dim, bias=bias
        )
        self.decoder_dropout = torch.nn.Dropout(dropouts[2])

    def encode(self, inputs, state=None):
        """Encode ``[batch, time, input_dim]`` and return all activities."""
        activities = []
        for step in range(inputs.shape[1]):
            activity, state = self.ctrnn_cell(inputs[:, step], state)
            activities.append(activity)

        if not activities:
            raise ValueError("CTRNN input sequence must contain at least one step")
        return torch.stack(activities, dim=1), state

    def decode(self, activity):
        return self.decoder_dropout(self.decoder_lin(activity))

    def forward(self, inputs, hidden=None):
        """Return predictions, all activities, and the final CTRNN state."""
        hidden_all, hidden_last = self.encode(inputs, hidden)
        output = self.decode(hidden_all)
        return output, hidden_all, hidden_last
