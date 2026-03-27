import torch
import triton
from kernels.rope import triton_rope

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["S"],
        x_vals=[512, 1024, 2048, 4096],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton RoPE", "PyTorch Baseline"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="rope",
        args={"B": 4, "H": 8, "D": 64},
    )
])
def bench_rope(B, H, S, D, provider):
    x = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    half = D // 2
    positions = torch.arange(S, device="cuda").float()
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device="cuda").float() / half))
    angles = torch.outer(positions, freqs)
    cos, sin = angles.cos(), angles.sin()
    if provider == "triton":
        fn = lambda: triton_rope(x, cos, sin)
    else:
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        x_real, x_imag = x[..., :half], x[..., half:]
        fn = lambda: torch.cat([x_real * cos_b - x_imag * sin_b,
                                 x_real * sin_b + x_imag * cos_b], dim=-1)
    return triton.testing.do_bench(fn)

if __name__ == "__main__":
    bench_rope.run(print_data=True, save_path="results")
