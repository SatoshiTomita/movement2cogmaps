from typing import Literal, Union

import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
from networks.activations import Activation
from utils.utils import ReTanh, re_tanh
# from vector_quantize_pytorch import LFQ


class CellBase(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        init_from_: Union[int, str] = "zeros",
        apply_tanh: bool = True,
    ) -> None:
        super().__init__()
        
        self.__hidden_dim = hidden_dim

        if isinstance(init_from_, int):
            print("init from obs")
            initializer_hidden_dim = (
                init_from_ if init_from_ > hidden_dim else hidden_dim
            )
            # print(init_from_obs)
            self.initializer = nn.Sequential(
                nn.Linear(init_from_, initializer_hidden_dim),
                nn.Mish(),
                nn.Linear(initializer_hidden_dim, hidden_dim),
                nn.Tanh() if apply_tanh else nn.Identity(),
            )
            self.initialize = self.initializer
        elif init_from_ == "zeros":
            self.initialize = self.zero_init
        elif init_from_ == "param":
            self.initial = nn.parameter.Parameter(torch.zeros([1, hidden_dim]))
            self.initialize = self.param_init
        else:
            raise NotImplementedError(f"init_from_ must be 'zeros' or 'param', got {type(int)}: {init_from_}")

    def zero_init(self, batch_size):
        return torch.zeros([batch_size, self.__hidden_dim]).to(self.device)

    def obs_init(self, obs):
        return self.initializer(obs)

    def param_init(self, batch_size):
        return self.initial.repeat(batch_size, 1)

    def detach(self):
        pass


class RNNCell(CellBase):
    def __init__(
        self, input_dim: int, hidden_dim: int, init_from_: int = 0, **kwargs
    ) -> None:
        super().__init__(hidden_dim, init_from_)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.rnn = nn.RNNCell(input_dim, hidden_dim, **kwargs)

        self.obs_dim = init_from_

    def init_latent(self, init_trigger, device=None):
        hidden_state = self.initialize(init_trigger)

        return hidden_state

    def forward(self, inputs, prev_hidden):
        hidden_state = self.rnn(inputs, prev_hidden)

        return hidden_state


class GRUCell(CellBase):
    def __init__(
        self, input_dim: int, hidden_dim: int, init_from_: int = 0, **kwargs
    ) -> None:
        super().__init__(hidden_dim, init_from_)

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.rnn = nn.GRUCell(input_dim, hidden_dim, **kwargs)

        self.obs_dim = init_from_

    def init_latent(self, init_trigger, device=None):
        hidden_state = self.initialize(init_trigger)

        return hidden_state

    def forward(self, inputs, prev_hidden):
        hidden_state = self.rnn(inputs, prev_hidden)

        return hidden_state



class SparseGateL0RDCell(CellBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        init_from_: int = "zeros",
        activation: str = "Mish",
        layers: int = 1,
        n_class: int = 4,
        dense_hidden_dim: int = 4,
        fix_sigma: bool = False,
        straight_through: bool = False,
        common_gate: bool = False,
        apply_softmax: bool = False,
        hidden_recurrence: bool = True,
    ) -> None:
        super().__init__(hidden_dim*n_class, init_from_)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.fix_sigma = fix_sigma
        self.common_gate = common_gate
        self.dense_hidden_dim = dense_hidden_dim
        self.hidden_recurrence = hidden_recurrence

        self.activation = getattr(nn, activation)()
        self.layers = layers
        self.gate_net = self._build_gate()
        self.proposal = self._build_proposal()
        self.retanh = ReTanh.apply if straight_through else re_tanh
        self.apply_softmax = apply_softmax

    def forward(self, inputs: torch.Tensor, prev_hidden: torch.Tensor):
        if self.hidden_recurrence:
            try:
                inputs = torch.cat([inputs, prev_hidden, self.hidden_state], dim=-1)
            except RuntimeError:
                hidden_state = self.hidden_state.detach().clone()
                inputs = torch.cat([inputs, prev_hidden, hidden_state.expand_as(prev_hidden)], dim=-1)
        else:
            inputs = torch.cat([inputs, prev_hidden], dim=-1)
        gate = self._gate(self.gate_net(inputs))
        self.hidden_state = gate * self.proposal(inputs) + (1 - gate) * self.hidden_state

        return self.argmax(self.hidden_state), gate

    def init_latent(self, init_trigger, device=None):
        self.hidden_state = self.initialize(init_trigger)
        return self.argmax(self.hidden_state)

    def argmax(self, input:torch.Tensor):
        shape = input.shape
        input = input.view(-1, self.n_class)
        if self.apply_softmax:
            input = mytorch.softmax(input, dim=-1)
        one_hot = F.one_hot(input.argmax(dim=-1), self.n_class)
        return one_hot.view(*shape) + input.view(*shape) - input.detach().view(*shape)

    def softplus(self, x):
        return F.softplus(x) + 1e-5

    def _gate(self, x):
        if self.fix_sigma:
            gate = x
            if self.training:
                gate = gate - torch.randn_like(gate) * self.fix_sigma
            binary_gate = self.retanh(gate)

        else:
            mu, std = torch.chunk(x, 2, dim=-1)
            std = self.softplus(std)
            if self.fix_sigma:
                std = torch.ones_like(std) * self.fix_sigma

            gate_dist = D.Normal(mu, std)
            gate_dist = D.Independent(gate_dist, 1)
            gate = gate_dist.rsample()
            binary_gate = self.retanh(gate)
        shape = binary_gate.shape
        return binary_gate.unsqueeze(-1).expand(*shape, self.n_class).flatten(-2, -1)

    def _build_gate(self):
        dense = []

        if self.hidden_recurrence:
            dense += [nn.Linear(self.input_dim +
                            self.hidden_dim*self.n_class*2, self.dense_hidden_dim)]
        else:
            dense += [nn.Linear(self.input_dim +
                            self.hidden_dim*self.n_class, self.dense_hidden_dim)]
        dense += [self.activation]

        for i in range(self.layers - 1):
            dense += [nn.Linear(self.dense_hidden_dim, self.dense_hidden_dim)]
            dense += [self.activation]

        if self.fix_sigma:
            dense += [nn.Linear(self.dense_hidden_dim, 
                                1 if self.common_gate else self.hidden_dim)]
        else:
            dense += [nn.Linear(self.dense_hidden_dim,
                                2 if self.common_gate else self.hidden_dim * 2)]

        return nn.Sequential(*dense)

    def _build_proposal(self):
        dense = []
        if self.hidden_recurrence:
            dense += [nn.Linear(self.input_dim + self.hidden_dim*self.n_class*2, self.dense_hidden_dim)]
        else:
            dense += [nn.Linear(self.input_dim + self.hidden_dim*self.n_class, self.dense_hidden_dim)]
        dense += [self.activation]

        for i in range(self.layers - 1):
            dense += [nn.Linear(self.dense_hidden_dim, self.dense_hidden_dim)]
            dense += [self.activation]
        
        dense += [nn.Linear(self.dense_hidden_dim, self.hidden_dim*self.n_class)]
        return nn.Sequential(*dense)

class GateL0RDCell(CellBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        init_from_: int = "zeros",
        activation: str = "Mish",
        layers: int = 1,
        fix_sigma: bool = False,
        straight_through: bool = False,
        common_gate: bool = False,
        dense_hidden_dim: int = None,
    ) -> None:
        super().__init__(hidden_dim, init_from_)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fix_sigma = fix_sigma
        self.common_gate = common_gate

        self.dense_hidden_dim = hidden_dim if dense_hidden_dim is None else dense_hidden_dim
        self.act_fn = getattr(nn, activation)()
        self.layers = layers
        self.gate_net = self._build_gate()
        self.proposal = self._build_proposal()
        self.retanh = ReTanh.apply if straight_through else re_tanh

    def forward(self, inputs, prev_hidden):
        inputs = torch.cat([inputs, prev_hidden], dim=-1)
        gate_hidden = self.gate_net(inputs)
        gate = self._gate(gate_hidden)
        hidden_state = gate * self.proposal(inputs) + (1 - gate) * prev_hidden

        return hidden_state, gate

    def init_latent(self, init_trigger, device=None):
        hidden_state = self.initialize(init_trigger)

        return hidden_state

    def softplus(self, x):
        return F.softplus(x) + 1e-5

    def _gate(self, x):
        if self.fix_sigma:
            gate = x
            if self.training:
                gate = gate - torch.randn_like(gate) * self.fix_sigma
            binary_gate = self.retanh(gate)

        else:
            mu, std = torch.chunk(x, 2, dim=-1)
            std = self.softplus(std)
            if self.fix_sigma:
                std = torch.ones_like(std) * self.fix_sigma

            gate_dist = D.Normal(mu, std)
            gate_dist = D.Independent(gate_dist, 1)
            gate = gate_dist.rsample()
            binary_gate = self.retanh(gate)

        return binary_gate

    def _build_gate(self):
        dense = []

        dense += [nn.Linear(self.input_dim +
                            self.hidden_dim, self.dense_hidden_dim)]

        for i in range(self.layers - 1):
            dense += [self.act_fn]
            dense += [nn.Linear(self.dense_hidden_dim, self.dense_hidden_dim)]

        if self.fix_sigma:
            dense += [nn.Identity()]
        else:
            dense += [self.act_fn]
            dense += [nn.Linear(self.dense_hidden_dim,
                                2 if self.common_gate else self.hidden_dim * 2)]

        return nn.Sequential(*dense)

    def _build_proposal(self):
        dense = []
        dense += [nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim)]

        for i in range(self.layers - 1):
            dense += [self.act_fn]
            dense += [nn.Linear(self.dense_hidden_dim, self.dense_hidden_dim)]

        # dense += [self.act_fn]
        # dense += [nn.Linear(self.dense_hidden_dim, self.hidden_dim)]
        dense += [nn.Tanh()]
        return nn.Sequential(*dense)


class MTRNNCell(CellBase):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        init_from_: int = 0,
        tau: int = 4,
        tau_sample: bool = False,
        bias: bool = True,
        return_gate: bool = False,
        apply_tanh: bool = True,
    ) -> None:
        super().__init__(hidden_dim, init_from_, apply_tanh)
        self.apply_tanh = apply_tanh

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.__d2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.__input2h = nn.Linear(input_dim, hidden_dim, bias=bias)

        if tau_sample:
            sampled_tau = D.Normal(tau, 1).sample([hidden_dim])
            sampled_tau = torch.clamp(sampled_tau, 1.0 + 1e-3)
            self.tau = nn.parameter.Parameter(sampled_tau, requires_grad=False)
            print("sampled tau", self.tau)

        else:
            self.tau = tau
            assert self.tau > 1.0

        self.init_from = init_from_
        self.return_gate = return_gate

    def detach(self):
        self.hidden = self.hidden.detach()



    def init_latent(self, init_trigger, device=None):
        print("init_trigger", init_trigger.shape)
        self.hidden = self.initialize(init_trigger)

        d = self.hidden
        if self.init_from == "param" or not self.apply_tanh:
            d = d.tanh()
        if not self.training:
            self.hidden_histopy = []

        return d

    def __compute_mtrnn(self, input, prev_d):
        # consider the input from a higher layer as well
        if not self.training:
            self.hidden = self.hidden.to(input.device)
            self.hidden_histopy.append(self.hidden)
        self.hidden = (1 - 1 / self.tau) * self.hidden + (
            self.__d2h(prev_d) + self.__input2h(input)
        ) / self.tau
        d = torch.tanh(self.hidden)
        return d

    def forward(self, inputs, prev_d):
        # print(self.hidden.shape)
        d_state = self.__compute_mtrnn(inputs, prev_d)

        if self.return_gate:
            return d_state, None
        else:
            return d_state

# class VQMTRNNCell(CellBase):
#     def __init__(
#         self,
#         input_dim: int,
#         hidden_dim: int,
#         init_from_: int = 0,
#         tau: int = 4,
#         tau_sample: bool = False,
#         bias: bool = True,
#         n_groups: int = 1,
#         codebook_size: int = 8,
#         w_commit: float = 1.0,
#         w_entropy: float = 0.0,
#         return_gate: bool = False,
#         apply_tanh: bool = True,
#     ) -> None:
#         super().__init__(hidden_dim, init_from_, apply_tanh)

#         self.hidden_dim = hidden_dim
#         self.input_dim = input_dim
#         self.n_groups = n_groups
#         self.codebook_size = codebook_size
#         self.apply_tanh = apply_tanh

#         self._d2h = nn.Linear(hidden_dim, hidden_dim, bias=bias)
#         self._input2h = nn.Linear(input_dim, hidden_dim, bias=bias)
#         self.quantizer = LFQ(
#             dim=hidden_dim // n_groups,
#             codebook_size=2**codebook_size,
#             spherical=True,
#             commitment_loss_weight=w_commit,
#             entropy_loss_weight=w_entropy,
#         )

#         if tau_sample:
#             sampled_tau = D.Normal(tau, 1).sample([hidden_dim])
#             sampled_tau = torch.clamp(sampled_tau, 1.0 + 1e-3)
#             self.tau = nn.parameter.Parameter(sampled_tau, requires_grad=False)
#             print("sampled tau", self.tau)

#         else:
#             self.tau = tau
#             assert self.tau > 1.0

#         self.init_from = init_from_
#         self.return_gate = return_gate

#     def init_latent(self, init_trigger, device=None):
#         hidden = self.initialize(init_trigger)
#         hidden = hidden.chunk(self.n_groups, dim=-1)
#         hidden = torch.stack(hidden, dim=-2)
#         quantized, _, vq_loss = self.quantizer(hidden)
#         self.hidden = quantized.flatten(-2, -1)

#         d = self.hidden
#         if self.init_from == "param" or not self.apply_tanh:
#             d = d.tanh()

#         return d

#     def _compute_mtrnn(self, input, prev_d):
#         # consider the input from a higher layer as well

#         hidden = (1 - 1 / self.tau) * self.hidden + (
#             self._d2h(prev_d) + self._input2h(input)
#         ) / self.tau
#         hidden = hidden.chunk(self.n_groups, dim=-1)
#         hidden = torch.stack(hidden, dim=-2)
#         quantized, _, vq_loss = self.quantizer(hidden)
#         self.hidden = quantized.flatten(-2, -1)
#         d = torch.tanh(self.hidden)
#         return d, vq_loss

#     def forward(self, inputs, prev_d):
#         d_state, vq_loss = self._compute_mtrnn(inputs, prev_d)

#         if self.return_gate:
#             return d_state, vq_loss, None
#         else:
#             return d_state, vq_loss
