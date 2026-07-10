"""
Project B benchmark: PyTorch eager vs torch.compile vs custom CUDA kernels.

Run after building the extension:
    python setup.py build_ext --inplace
    python benchmark.py
"""

import torch
import torch.utils.benchmark as benchmark

import activations_cuda

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE  = torch.float32    # TODO: also try torch.float16 (half precision)
SIZES  = [1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22]  # 16K → 4M elements

WARMUP_ITERS = 5
BENCH_ITERS  = 50

# ---------------------------------------------------------------------------
# PyTorch reference implementations — x is always (N,), a flat 1-D tensor
# ---------------------------------------------------------------------------

def pt_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def pt_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)

def pt_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)

# ---------------------------------------------------------------------------
# torch.compile wrappers — first call triggers compilation, warm up before timing
# ---------------------------------------------------------------------------

compiled_relu = torch.compile(pt_relu)
compiled_silu = torch.compile(pt_silu)
compiled_gelu = torch.compile(pt_gelu)

# ---------------------------------------------------------------------------
# Custom kernel wrappers
# ---------------------------------------------------------------------------

def custom_relu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.relu_fwd(x)

def custom_silu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.silu_fwd(x)

def custom_gelu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.gelu_fwd(x)

# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------

def check_correctness(name: str, ref_fn, custom_fn, x: torch.Tensor) -> None:
    """Compare custom kernel output against PyTorch reference."""
    with torch.no_grad():
        ref = ref_fn(x)
        out = custom_fn(x)
        if (torch.allclose(ref, out, atol=1e-5)):
            print("PASS")
        else:
            print("FAIL")

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_fn(fn, x: torch.Tensor, label: str) -> float:
    """Returns median latency in milliseconds."""
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            fn(x)
    torch.cuda.synchronize()

    # TODO: time BENCH_ITERS runs (benchmark.Timer or CUDA events), return median ms
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Per-kernel benchmark
# ---------------------------------------------------------------------------

def bench_relu() -> None:
    print("\n=== ReLU ===")
    print(f"{'N':>10}  {'eager_ms':>10}  {'compile_ms':>12}  {'custom_ms':>10}  {'speedup':>8}")
    for N in SIZES:
        x = None  # TODO: torch.randn(N, device=DEVICE, dtype=DTYPE)

        t_eager   = time_fn(pt_relu,       x, "eager")
        t_compile = time_fn(compiled_relu, x, "compile")
        t_custom  = time_fn(custom_relu,   x, "custom")

        speedup = t_eager / t_custom
        print(f"{N:>10}  {t_eager:>10.3f}  {t_compile:>12.3f}  {t_custom:>10.3f}  {speedup:>7.2f}x")

def bench_silu() -> None:
    # TODO: same structure as bench_relu but for SiLU
    print("\n=== SiLU ===")
    raise NotImplementedError

def bench_gelu() -> None:
    # TODO: same structure as bench_relu but for GELU
    print("\n=== GELU ===")
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Bandwidth calculation
# ---------------------------------------------------------------------------

def compute_bandwidth_gb_s(N: int, dtype: torch.dtype, time_ms: float) -> float:
    # TODO: bytes_moved = 2 * N * sizeof(dtype); return bytes_moved / (time_s * 1e9)
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype:  {DTYPE}")

    # TODO: check_correctness for each kernel before benchmarking

    bench_relu()
    bench_silu()
    bench_gelu()
