import torch
import pytest
from kernels.int8_matmul import triton_int8_matmul

def test_int8_matmul():
    M, K, N = 256, 128, 256
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")
    ref = torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.int32)
    out = triton_int8_matmul(a, b)
    # atol=128 accounts for fp16 rounding accumulated over K=128 steps
    torch.testing.assert_close(out.float(), ref.float(), atol=128, rtol=0)
