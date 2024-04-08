from jaxtyping import Array, Float32
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Literal
from .rmsnorm import RMSNorm


class Gated_MLP_block(nn.Module):
    def __init__(self, input_dim:int, expansion_factor: int=3,
                 approximate:Literal['none', 'tanh']='none') -> None:
        super().__init__()
        self.D = input_dim
        self.M = expansion_factor
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=self.D, out_features=self.D*self.M)
        self.p2 = nn.Linear(in_features=self.D, out_features=self.D*self.M)
        self.p3 = nn.Linear(in_features=self.D*self.M, out_features=self.D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1, self.approximate)

        # right branch
        x2 = self.p2(x)

        y = x1*x2
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    #TODO -
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:Tensor) -> Tensor:
        return


class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(self, dim:int, expansion_factor:int|float=3,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.input_dim = dim
        self.hidden_dim = int(round(dim * expansion_factor))

        self.Wa = nn.Parameter(torch.empty(self.hidden_dim, dim, **factory_kwargs))
        self.Wx = nn.Parameter(torch.empty(self.hidden_dim, dim, **factory_kwargs))
        self.ba = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.bx = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.Lambda = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))  # Λ
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://tinyurl.com/lecuninit
        nn.init.normal_(self.Wa, mean=0, std=1 / (self.input_dim ** 0.5))
        nn.init.normal_(self.Wx, mean=0, std=1 / (self.input_dim ** 0.5))

        # init bias

        # init Λ
        nn.init.uniform_(self.Lambda, a=0.9, b=0.999)
        self.Lambda = - torch.log(
            (self.Lambda ** (-1./self.c) ) - 1.
        )

    def forward(self, x:Float32[Array, "batch_size, sequence_length, dim"]
                ) -> Float32[Array, "batch_size, sequence_length, dim"]:

        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim, dtype=self.dtype, device=self.device)
        y = []
        for t in range(sequence_length):
            xt = x[:, t, :]
            rt = torch.sigmoid(F.linear(xt, self.Wa, self.ba))  # (1)
            it = torch.sigmoid(F.linear(xt, self.Wx, self.bx))  # (2)

            # TODO (3)
            # a = torch.sigmoid(self.Lambda)
            # at = torch.pow(a, self.c*rt)

            # TODO Appendix A - https://github.com/kyegomez/Griffin/issues/6
            log_at = - self.c * F.softplus(-self.Lambda, beta=1, threshold=20) * rt # (6)
            at = torch.exp(log_at)

            ht = at * ht + torch.sqrt(1 - at**2) * (it * xt) # (4)

            y.append(ht.unsqueeze(1))

        y = torch.cat(y, dim=1)

        return y


class RGLRU(Real_Gated_Linear_Recurrent_Unit): ...


class Recurrent_block(nn.Module):
    def __init__(self, input_dim:int, D_rnn:int=...,
                 approximate:Literal['none', 'tanh']='none'):
        super().__init__()
        self.D = input_dim
        self.D_rnn = D_rnn
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=self.D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=self.D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=..., out_features=self.D)


    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1, self.approximate)

        # right branch
        x2 = self.p2(x)

        return


class Residual_block(nn.Module):
    def __init__(self, input_dim:int):
        super().__init__()
        self.mlp = Gated_MLP_block(input_dim, expansion_factor=3)
        self.tmb = Recurrent_block(input_dim, D_rnn=...)
        self.rmsnorm = RMSNorm(d=input_dim) # ?

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.rmsnorm(x)
        x1 = self.tmb(x1)

        x = x + x1

        x2 = self.rmsnorm(x)
        x2 = self.mlp(x2)

        x = x + x2

        return x
