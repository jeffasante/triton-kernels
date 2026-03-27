import torch
import pytest
from kernels.rope import triton_rope

def test_rope():
    B, S, H, D = 2, 64, 8, 64
    x = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    half = D // 2
    positions = torch.arange(S, device="cuda").float()
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device="cuda").float() / half))
    angles = torch.outer(positions, freqs)
    cos = angles.cos()  # [S, half]
    sin = angles.sin()  # [S, half]

    x_real, x_imag = x[..., :half], x[..., half:]
    # reshape cos/sin to [1, S, 1, half] so it broadcasts over B and H
    cos_b = cos.unsqueeze(0).unsqueeze(2)  # [1, S, 1, half]
    sin_b = sin.unsqueeze(0).unsqueeze(2)  # [1, S, 1, half]
    ref = torch.cat([x_real * cos_b - x_imag * sin_b,
                     x_real * sin_b + x_imag * cos_b], dim=-1)

    out = triton_rope(x, cos, sin)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
