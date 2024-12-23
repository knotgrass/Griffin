from jaxtyping import Array, Float32
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Literal
from .rmsnorm import RMSNorm


class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: Literal["none", "tanh"] = "none",
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p2 = nn.Linear(in_features=D, out_features=D*self.M)
        self.p3 = nn.Linear(in_features=D*self.M, out_features=D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    def __init__(self, D: int, kernel_size: int=4):
        super().__init__()
        # A separable 1D convolution:
        # - Input channels = output channels = D
        # - groups = D makes it depthwise (channel-wise) convolution.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            bias=False,
            padding=kernel_size // 2  # optional, to preserve sequence length
        )

    def forward(self, x: Tensor) -> Tensor:
        # https://chatgpt.com/share/67692a55-5224-8005-a271-80067aa3bcbb
        # B = Batch size
        # T = Sequence length
        # D = Feature dimension
        # x: (B, T, D)
        # Transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)  # (B, D, T)
        # Transpose back to (B, T, D)
        x = x.transpose(1, 2)
        return x


class Real_Gated_Linear_Recurrent_Unit(nn.Module):
    c = 8.0

    def __init__(
        self, D: int, expansion_factor: int | float = 3, device=None, dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dtype = dtype
        self.device = device
        self.D = D
        self.hidden_dim = int(round(D * expansion_factor))

        self.Wa = nn.Parameter(torch.empty(self.hidden_dim, D, **factory_kwargs))
        self.Wx = nn.Parameter(torch.empty(self.hidden_dim, D, **factory_kwargs))
        self.ba = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.bx = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))
        self.Lambda = nn.Parameter(torch.empty(self.hidden_dim, **factory_kwargs))  # Λ
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # https://tinyurl.com/lecuninit
        nn.init.normal_(self.Wa, mean=0, std=1 / (self.D ** 0.5))
        nn.init.normal_(self.Wx, mean=0, std=1 / (self.D ** 0.5))

        # init bias

        # init Λ
        nn.init.uniform_(self.Lambda, a=0.9, b=0.999)
        self.Lambda = - torch.log(
            (self.Lambda ** (-1./self.c) ) - 1.
        )

    def foresee(self, x:Float32[Array, "batch_size, sequence_length, dim"]
                ) -> Float32[Array, "batch_size, sequence_length, dim"]:

        batch_size, sequence_length = x.shape[:2]
        ht = torch.zeros(batch_size, self.hidden_dim,
                         dtype=self.dtype, device=self.device)
        y = torch.empty(batch_size, sequence_length, self.hidden_dim,
                        dtype=self.dtype, device=self.device)
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

            y[:, t, :] = ht

        return y

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
    def __init__(self, D:int, D_rnn:int=...,
                 approximate:Literal['none', 'tanh']='none'):
        super().__init__()
        self.D = D
        self.D_rnn = D_rnn
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        self.separableConv1D = Temporal_Conv1D(D, kernel_size=4)
        self.rglru = RGLRU(self.D)

    def forward(self, x:Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)
        x2 = self.separableConv1D(x2)
        x2 = self.rglru(x2)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)
        return y


class Residual_block(nn.Module):
    def __init__(self, D:int):
        super().__init__()
        self.mlp = Gated_MLP_block(D, expansion_factor=3)
        self.tmb = Recurrent_block(D, D_rnn=int(4*D/3))
        self.rmsnorm = RMSNorm(d=D) # ?

    def forward(self, x:Tensor) -> Tensor:
        x1 = self.rmsnorm(x)
        x1 = self.tmb(x1)

        y1 = x + x1

        x2 = self.rmsnorm(y1)
        x2 = self.mlp(x2)

        y2 = y1 + x2

        return y2
