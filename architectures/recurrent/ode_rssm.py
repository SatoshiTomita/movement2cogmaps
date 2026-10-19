"""ODE-RSSM: an RSSM whose deterministic state follows a learned neural ODE.

Within one data step the input ``x = [action_t, z_(t-1)]`` is held constant
(zero-order hold) and the membrane potential ``u`` is integrated over the
unit interval with a fixed-step solver::

    tau * du/dt = -u + F(h, x),    h = activation(u)
    F(h, x)     = W2 tanh(W1 [h, x] + b1) + b2

``F`` is a learned vector field (an MLP), so the dynamics are not restricted
to the leaky linear-recurrent form of the CTRNN.  With a linear ``F`` and one
Euler sub-step the update reduces exactly to :class:`CTRNNCell`.  The
stochastic state, prior/posterior, initialisation and grid-cell input are
inherited unchanged from :class:`CTRSSMPredictor`.
"""

import torch
from torch import nn

from architectures.recurrent.ctrssm import CTRSSMPredictor
from architectures.rnn_core import ACTIVATIONS


class ODECell(nn.Module):
    """One data step of the neural-ODE deterministic transition."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        tau=4.0,
        nonlinearity="sigmoid",
        dropouts=(0, 0),
        substeps=4,
        method="rk4",
        field_hidden_dim=None,
    ):
        super().__init__()
        if tau < 1.0:
            raise ValueError(f"ODE-RSSM tau must be at least 1 data step, but got {tau}")
        if substeps < 1:
            raise ValueError(f"ODE-RSSM needs at least one sub-step, but got {substeps}")
        if method not in {"rk4", "euler"}:
            raise ValueError(f"Unknown ODE solver {method!r}; choose 'rk4' or 'euler'")
        if dropouts[1]:
            # A recurrent mask would differ between solver evaluations.
            raise ValueError("ODE-RSSM does not support recurrent dropout")
        if nonlinearity not in ACTIVATIONS:
            raise ValueError(
                f"Unknown ODE-RSSM nonlinearity {nonlinearity!r}; "
                f"choose one of {sorted(ACTIVATIONS)}"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.substeps = int(substeps)
        self.method = method
        field_hidden_dim = field_hidden_dim or hidden_dim
        self.activation_fn = ACTIVATIONS[nonlinearity](hidden_dim)
        self.in2field = nn.Linear(input_dim, field_hidden_dim)
        self.hidden2field = nn.Linear(hidden_dim, field_hidden_dim, bias=False)
        self.field2membrane = nn.Linear(field_hidden_dim, hidden_dim)
        self.in_dropout = nn.Dropout(dropouts[0])
        self.register_buffer("tau", torch.tensor(float(tau)))

    def initial_state(self, inputs):
        shape = (inputs.shape[0], self.hidden_dim)
        return inputs.new_zeros(shape), inputs.new_zeros(shape)

    def vector_field(self, membrane, input_drive):
        """Return du/dt; ``input_drive`` is the constant input projection."""
        activity = self.activation_fn(membrane)
        field = self.field2membrane(
            torch.tanh(input_drive + self.hidden2field(activity))
        )
        return (field - membrane) / self.tau

    def forward(self, inputs, state=None):
        """Return ``(activity, (membrane, activity))`` after one data step."""
        if state is None:
            membrane, _ = self.initial_state(inputs)
        else:
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError("ODE-RSSM state must be a (membrane, activity) tuple")
            membrane, _ = state

        # The input is held constant over the step, so project it only once.
        input_drive = self.in2field(self.in_dropout(inputs))
        dt = 1.0 / self.substeps
        for _ in range(self.substeps):
            k1 = self.vector_field(membrane, input_drive)
            if self.method == "euler":
                membrane = membrane + dt * k1
                continue
            k2 = self.vector_field(membrane + 0.5 * dt * k1, input_drive)
            k3 = self.vector_field(membrane + 0.5 * dt * k2, input_drive)
            k4 = self.vector_field(membrane + dt * k3, input_drive)
            membrane = membrane + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        activity = self.activation_fn(membrane)
        return activity, (membrane, activity)


class ODERSSMPredictor(CTRSSMPredictor):
    """CTRSSM with the Euler CTRNN transition replaced by :class:`ODECell`."""

    def __init__(self, *args, ode_substeps=4, ode_method="rk4", **kwargs):
        super().__init__(*args, **kwargs)
        cell = self.ctrnn_cell
        dropouts = kwargs.get("dropouts", (0.0, 0.0, 0.0))
        self.ctrnn_cell = ODECell(
            input_dim=cell.input_dim,
            hidden_dim=cell.hidden_dim,
            tau=float(cell.tau),
            nonlinearity=kwargs.get("nonlinearity", "sigmoid"),
            dropouts=dropouts[:2],
            substeps=ode_substeps,
            method=ode_method,
        )
