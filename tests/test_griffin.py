import torch

from griffin.griffin import Recurrent_block, Residual_block, Temporal_Conv1D


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


def test_residual_block():
    """Residual_block runs with D_rnn != D and keeps the fixes of C3-C9 and S1-S3."""
    torch.manual_seed(0)
    B, T, D = 2, 10, 6  # D_rnn = 8 != D, so a D / D_rnn mix-up fails
    blk = Residual_block(D)
    rglru = blk.tmb.rglru

    # C3: Lambda stays an nn.Parameter
    assert isinstance(rglru.Lambda, torch.nn.Parameter)
    # C4: Recurrent_block(D) without D_rnn gets the same width
    assert Recurrent_block(D).D_rnn == blk.tmb.D_rnn == 8
    # S2: reset_parameters() sets the gate biases to 0, whatever was in memory
    with torch.no_grad():
        rglru.ba.fill_(1.0)
        rglru.bx.fill_(1.0)
    rglru.reset_parameters()
    assert (rglru.ba == 0).all() and (rglru.bx == 0).all()
    # S3: one RMSNorm per branch
    assert blk.tmb_norm is not blk.mlp_norm

    # C5-C8: forward runs and keeps the shape
    x = torch.randn(B, T, D, requires_grad=True)
    y = blk(x)
    assert y.shape == (B, T, D)

    # S1: y[t] never uses x[s] with s > t
    for t in range(T):
        (grad,) = torch.autograd.grad(y[:, t].sum(), x, retain_graph=True)
        assert grad[:, t + 1:].abs().sum() == 0, f"y[{t}] uses a future step"

    y.sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name

    # C9: h_0 follows the dtype and the device of x
    blk_bf16 = Residual_block(D).to(torch.bfloat16)
    assert blk_bf16(x.detach().to(torch.bfloat16)).dtype == torch.bfloat16
    # the meta device holds no data, so it checks the device without a GPU
    blk_meta = Residual_block(D).to("meta")
    assert blk_meta(x.detach().to("meta")).device.type == "meta"
