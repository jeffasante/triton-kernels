import torch
import triton
from kernels.attention import triton_attention

def baseline_attention(q, k, v, causal=True):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).half()

configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton Fused", "PyTorch Baseline"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="attention-fwd",
        args={"B": 4, "H": 8, "D": 64},
    )
]

@triton.testing.perf_report(configs)
def bench_attention(B, H, N_CTX, D, provider):
    q = torch.randn(B, H, N_CTX, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N_CTX, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N_CTX, D, device="cuda", dtype=torch.float16)
    if provider == "triton":
        fn = lambda: triton_attention(q, k, v)
    else:
        fn = lambda: baseline_attention(q, k, v)
    ms = triton.testing.do_bench(fn)
    return ms

if __name__ == "__main__":
    bench_attention.run(print_data=True, save_path="results")
