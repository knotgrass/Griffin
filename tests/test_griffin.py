import torch

from griffin.griffin import Temporal_Conv1D


def test_causal():
    """Temporal_Conv1D keeps the length T, and y[t] never uses x[s] with s > t."""
    torch.manual_seed(0)
    B, T, D = 2, 10, 8
    conv = Temporal_Conv1D(D, kernel_size=4)
    x = torch.randn(B, T, D, requires_grad=True)

    y = conv(x)
    assert y.shape == (B, T, D)

    for t in range(T):
        # grad[:, s] = d(sum of y[:, t]) / d(x[:, s]): non-zero only if y[t] uses x[s].
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        assert grad[:, t + 1:].abs().sum() == 0, f"y[{t}] uses a future step"
        assert grad[:, t].abs().sum() > 0, f"y[{t}] ignores the current step"
