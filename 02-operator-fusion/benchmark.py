"""
Project B benchmark: PyTorch eager vs torch.compile vs custom CUDA kernels.

Three-way comparison for each activation function:
  1. PyTorch eager  — standard torch ops, no compilation
  2. torch.compile  — XLA/Inductor compilation (run once to warm, then bench)
  3. Custom kernel  — our hand-written CUDA kernel via activations_cuda

Run after building the extension:
    python setup.py build_ext --inplace
    python benchmark.py

Key Python/PyTorch concepts used here:
  - torch.Tensor creation: torch.randn, torch.zeros, .to(device), .half()
  - torch.cuda.synchronize()   — flush GPU work before timing
  - torch.compile(fn)          — wraps a callable, returns optimized version
  - torch.utils.benchmark.Timer — accurate GPU timing with warmup + stats
  - context managers: with torch.no_grad(): ...
"""

import csv
import torch
import torch.utils.benchmark as benchmark
import numpy as np

import activations_cuda

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
# TODO: also try torch.float16 (half precision)
DTYPE = torch.float32
# 16K → 4M elements
SIZES = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]
C = 1024
EPS = 1e-5

WARMUP_ITERS = 5
BENCH_ITERS  = 50

# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------
# These are the "ground truth" we benchmark against.
# Familiarise yourself with the shapes: x is always (N,) — a flat 1-D tensor.

def pt_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def pt_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x + bias)

def pt_add_layernorm(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.layer_norm(x + residual, (x.shape[-1],), gamma, beta, EPS)

# ---------------------------------------------------------------------------
# torch.compile wrappers
# ---------------------------------------------------------------------------

compiled_relu = torch.compile(pt_relu)
compiled_bias_gelu = torch.compile(pt_bias_gelu)
compiled_add_layernorm = torch.compile(pt_add_layernorm)

# ---------------------------------------------------------------------------
# Custom kernel wrappers
# ---------------------------------------------------------------------------

def custom_relu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.relu_fwd(x)

def custom_bias_gelu(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return activations_cuda.bias_gelu_fwd(x, bias)

def custom_add_layernorm(x: torch.Tensor, residual: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return activations_cuda.add_layernorm_fwd(x, residual, gamma, beta, EPS)

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------
# Before benchmarking, verify your kernels produce the same result as PyTorch.
# torch.allclose(a, b, atol=1e-5) returns True if all elements match.

def check_correctness(name: str, ref_fn, custom_fn, x: torch.Tensor) -> None:
    """Compare custom kernel output against PyTorch reference."""
    with torch.no_grad():
        ref = ref_fn(x)
        out = custom_fn(x)
        if (torch.allclose(ref, out, atol=1e-5)):
            print(f"{name} passes")
        else:
            print(f"{name} fails")


@torch.inference_mode()
def time_fn(fn, x: torch.Tensor) -> float:
    # Warmup
    for _ in range(WARMUP_ITERS):
        fn(x)
    torch.cuda.synchronize()

    elapsed_times = []

    for _ in range(BENCH_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(x)
        end.record()

        end.synchronize()

        ms = start.elapsed_time(end)
        elapsed_times.append(ms)

    return float(np.median(elapsed_times))

# ---------------------------------------------------------------------------
# Per-kernel benchmark
# ---------------------------------------------------------------------------

def bench_relu() -> list:
    print("\n=== ReLU ===")
    print(f"{'N':>10}  {'eager_ms':>10}  {'compile_ms':>12}  {'custom_ms':>10}  {'speedup':>8}")
    b = torch.tensor([], dtype=DTYPE).element_size()
    records = []
    for N in SIZES:
        x = torch.randn(N, device=DEVICE, dtype=DTYPE)

        t_eager   = time_fn(pt_relu,       x)
        t_compile = time_fn(compiled_relu, x)
        t_custom  = time_fn(custom_relu,   x)

        # how much faster than eager?
        speedup = t_eager / t_custom
        print(f"{N:>10}  {t_eager:>10.3f}  {t_compile:>12.3f}  {t_custom:>10.3f}  {speedup:>7.2f}x")

        bytes_moved = 2 * N * b
        for variant, t in (("eager", t_eager), ("compile", t_compile), ("custom", t_custom)):
            records.append(("relu", variant, N, t, compute_bandwidth_gb_s(bytes_moved, t)))
    return records

def bench_bias_gelu() -> list:
    print("\n=== Bias + GELU (fused) ===")
    print(f"{'N':>10}  {'eager_ms':>10}  {'compile_ms':>12}  {'custom_ms':>10}  {'speedup':>8}")
    b = torch.tensor([], dtype=DTYPE).element_size()
    records = []
    for N in SIZES:
        rows = N // C
        x = torch.randn(rows, C, device=DEVICE, dtype=DTYPE)
        bias = torch.randn(C, device=DEVICE, dtype=DTYPE)

        t_eager = time_fn(lambda t: pt_bias_gelu(t, bias), x)
        t_compile = time_fn(lambda t: compiled_bias_gelu(t, bias), x)
        t_custom = time_fn(lambda t: custom_bias_gelu(t, bias), x)

        speedup = t_eager / t_custom
        print(f"{N:>10}  {t_eager:>10.3f}  {t_compile:>12.3f}  {t_custom:>10.3f}  {speedup:>7.2f}x")

        bytes_moved = (2 * N + C) * b
        for variant, t in (("eager", t_eager), ("compile", t_compile), ("custom", t_custom)):
            records.append(("bias_gelu", variant, N, t, compute_bandwidth_gb_s(bytes_moved, t)))
    return records


def bench_add_layernorm() -> list:
    print("\n=== Add + LayerNorm (fused) ===")
    print(f"{'N':>10}  {'eager_ms':>10}  {'compile_ms':>12}  {'custom_ms':>10}  {'speedup':>8}")
    b = torch.tensor([], dtype=DTYPE).element_size()
    records = []
    for N in SIZES:
        rows = N // C
        x = torch.randn(rows, C, device=DEVICE, dtype=DTYPE)
        residual = torch.randn(rows, C, device=DEVICE, dtype=DTYPE)

        gamma = torch.randn(C, device=DEVICE, dtype=DTYPE)
        beta = torch.randn(C, device=DEVICE, dtype=DTYPE)

        t_eager = time_fn(lambda t: pt_add_layernorm(t, residual, gamma, beta), x)
        t_compile = time_fn(lambda t: compiled_add_layernorm(t, residual, gamma, beta), x)
        t_custom = time_fn(lambda t: custom_add_layernorm(t, residual, gamma, beta), x)

        speedup = t_eager / t_custom
        print(f"{N:>10}  {t_eager:>10.3f}  {t_compile:>12.3f}  {t_custom:>10.3f}  {speedup:>7.2f}x")

        bytes_moved = (3 * N + 2 * C) * b
        for variant, t in (("eager", t_eager), ("compile", t_compile), ("custom", t_custom)):
            records.append(("add_layernorm", variant, N, t, compute_bandwidth_gb_s(bytes_moved, t)))
    return records


# ---------------------------------------------------------------------------
# Bandwidth
# ---------------------------------------------------------------------------
# bytes_moved per op (b = sizeof(dtype)):
#   relu          -> 2*N * b
#   bias_gelu     -> (2*N + C) * b
#   add_layernorm -> (3*N + 2*C) * b

def compute_bandwidth_gb_s(bytes_moved: int, time_ms: float) -> float:
    return bytes_moved / (time_ms * 1e6)

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records: list, path: str = "results.csv") -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "variant", "N", "time_ms", "bw_gb_s"])
        w.writerows(records)
    print(f"Wrote {path} ({len(records)} rows)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype:  {DTYPE}")

    x_small = torch.randn(4, C, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(C, device=DEVICE, dtype=DTYPE)
    residual = torch.randn(4, C, device=DEVICE, dtype=DTYPE)
    gamma = torch.ones(C, device=DEVICE, dtype=DTYPE)
    beta = torch.zeros(C, device=DEVICE, dtype=DTYPE)

    check_correctness("relu", pt_relu, custom_relu, torch.randn(1024, device=DEVICE, dtype=DTYPE))
    check_correctness("bias_gelu", lambda t: pt_bias_gelu(t, bias), lambda t: custom_bias_gelu(t, bias), x_small)
    check_correctness("add_layernorm", lambda t: pt_add_layernorm(t, residual, gamma, beta), lambda t: custom_add_layernorm(t, residual, gamma, beta), x_small)

    records = []
    records += bench_relu()
    records += bench_bias_gelu()
    records += bench_add_layernorm()
    write_csv(records)
