import torch
import triton
import triton.language as tl

@triton.jit
def rope_kernel(
    x_ptr, out_ptr,
    cos_ptr, sin_ptr,
    seq_len, n_heads, head_dim,
    stride_xb, stride_xs, stride_xh, stride_xd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (batch, seq, head) triple
    b = pid // (seq_len * n_heads)
    s = (pid % (seq_len * n_heads)) // n_heads
    h = pid % n_heads

    half_dim = head_dim // 2
    offs = tl.arange(0, BLOCK_SIZE // 2)

    x_real = tl.load(x_ptr + b*stride_xb + s*stride_xs + h*stride_xh + offs)
    x_imag = tl.load(x_ptr + b*stride_xb + s*stride_xs + h*stride_xh + offs + half_dim)

    cos = tl.load(cos_ptr + s * half_dim + offs)
    sin = tl.load(sin_ptr + s * half_dim + offs)

    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos

    tl.store(out_ptr + b*stride_xb + s*stride_xs + h*stride_xh + offs, out_real)
    tl.store(out_ptr + b*stride_xb + s*stride_xs + h*stride_xh + offs + half_dim, out_imag)


def triton_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, S, H, D = x.shape
    out = torch.empty_like(x)
    grid = (B * S * H,)
    rope_kernel[grid](
        x, out, cos, sin,
        S, H, D,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_SIZE=D,
    )
    return out
