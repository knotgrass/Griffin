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
    #TODO   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x:Tensor) -> Tensor:
        return


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
