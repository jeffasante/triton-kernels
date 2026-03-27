import torch
import pytest
from kernels.attention import triton_attention

def reference_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(scores.shape[-2:], device=q.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).half()

@pytest.mark.parametrize("B,H,N,D", [(2, 4, 512, 64), (1, 8, 1024, 64)])
def test_attention_correctness(B, H, N, D):
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    ref = reference_attention(q, k, v)
    out = triton_attention(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
