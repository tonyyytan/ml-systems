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

import torch
import torch.utils.benchmark as benchmark

# TODO: uncomment once you've built the extension with setup.py
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
# PyTorch reference implementations
# ---------------------------------------------------------------------------
# These are the "ground truth" we benchmark against.
# Familiarise yourself with the shapes: x is always (N,) — a flat 1-D tensor.

def pt_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

def pt_silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)

def pt_gelu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x)

# ---------------------------------------------------------------------------
# torch.compile wrappers
# ---------------------------------------------------------------------------
# torch.compile(fn) returns a new callable that Inductor compiles to
# optimized GPU code (triton kernels) the first time it runs.
#
# Common modes:
#   torch.compile(fn)                          — default (inductor backend)
#   torch.compile(fn, mode="reduce-overhead")  — extra fusion passes
#   torch.compile(fn, fullgraph=True)          — error if graph breaks
#
# IMPORTANT: the first call triggers compilation (slow). Always warm up
# before timing.

# TODO: create compiled versions of each activation
# e.g.:   compiled_relu = torch.compile(pt_relu)
compiled_relu = torch.compile(pt_relu)
compiled_silu = torch.compile(pt_silu)
compiled_gelu = torch.compile(pt_gelu)

# ---------------------------------------------------------------------------
# Custom kernel wrappers
# ---------------------------------------------------------------------------
# Once you've registered ops in the PYBIND11_MODULE block of kernels.cu,
# call them here like normal Python functions.

def custom_relu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.relu_fwd(x)

def custom_silu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.silu_fwd(x)

def custom_gelu(x: torch.Tensor) -> torch.Tensor:
    return activations_cuda.gelu_fwd(x)

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
            print("PASS")
        else:
            print("FAIL")

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
# torch.utils.benchmark.Timer is the recommended way to time GPU ops.
# It handles:
#   - CUDA event timing (not wall-clock)
#   - warmup iterations
#   - statistical outlier trimming
#
# Example usage:
#
#   t = benchmark.Timer(
#       stmt="fn(x)",
#       globals={"fn": my_fn, "x": x},
#       num_threads=1,
#   )
#   result = t.timeit(number=BENCH_ITERS)
#   print(result.mean * 1e3, "ms")
#
# Alternatively, manual timing with CUDA events:
#
#   start = torch.cuda.Event(enable_timing=True)
#   end   = torch.cuda.Event(enable_timing=True)
#   start.record()
#   fn(x)
#   end.record()
#   torch.cuda.synchronize()
#   elapsed_ms = start.elapsed_time(end)

def time_fn(fn, x: torch.Tensor, label: str) -> float:
    """
    Returns median latency in milliseconds.
    TODO: implement using either benchmark.Timer or manual CUDA events.
    """
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            fn(x)
    torch.cuda.synchronize()

    # TODO: time BENCH_ITERS runs and return the median ms
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Per-kernel benchmark
# ---------------------------------------------------------------------------

def bench_relu() -> None:
    print("\n=== ReLU ===")
    print(f"{'N':>10}  {'eager_ms':>10}  {'compile_ms':>12}  {'custom_ms':>10}  {'speedup':>8}")
    for N in SIZES:
        # TODO: create input tensor of size N on DEVICE with DTYPE
        x = None  # e.g. torch.randn(N, device=DEVICE, dtype=DTYPE)

        t_eager   = time_fn(pt_relu,       x, "eager")
        t_compile = time_fn(compiled_relu, x, "compile")
        t_custom  = time_fn(custom_relu,   x, "custom")

        speedup = t_eager / t_custom  # how much faster than eager?
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
# Activation functions are memory-bound: each element is read once, written once.
# Achieved bandwidth = bytes_moved / time
#
# bytes_moved = 2 * N * sizeof(dtype)   (one read + one write)
# bw_gb_s     = bytes_moved / (time_s * 1e9)

def compute_bandwidth_gb_s(N: int, dtype: torch.dtype, time_ms: float) -> float:
    # TODO: compute and return achieved memory bandwidth in GB/s
    raise NotImplementedError

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"dtype:  {DTYPE}")

    # TODO: run correctness checks before benchmarking, e.g.:
    # x_small = torch.randn(1024, device=DEVICE, dtype=DTYPE)
    # check_correctness("relu", pt_relu, custom_relu, x_small)

    bench_relu()
    bench_silu()
    bench_gelu()
