import torch
import triton
from kernels.int8_matmul import triton_int8_matmul

@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=[256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["triton", "pytorch"],
        line_names=["Triton int8", "PyTorch fp32"],
        styles=[("blue", "-"), ("red", "--")],
        ylabel="ms",
        plot_name="int8-matmul",
        args={"N": 1024, "K": 1024},
    )
])
def bench_int8_matmul(M, N, K, provider):
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (K, N), dtype=torch.int8, device="cuda")
    if provider == "triton":
        fn = lambda: triton_int8_matmul(a, b)
    else:
        fn = lambda: torch.matmul(a.to(torch.float32), b.to(torch.float32))
    return triton.testing.do_bench(fn)

if __name__ == "__main__":
    bench_int8_matmul.run(print_data=True, save_path="results")
