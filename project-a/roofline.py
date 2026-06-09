"""
Project A: Roofline plot for square fp32 and fp16 matmul on RTX 5070 laptop GPU.

Goal: benchmark matmul at various sizes, compute arithmetic intensity,
and overlay measured TFLOPS on a roofline model.

Roofline model recap:
  - x-axis: arithmetic intensity (FLOPS / byte)
  - y-axis: attainable FLOPS/s
  - Two ceilings: memory bandwidth bound, compute bound
  - Ridge point: where the two ceilings intersect
"""

import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Hardware constants for RTX 5070 laptop GPU
#    TODO: fill these in from the spec sheet / nvidia-smi
# ---------------------------------------------------------------------------
PEAK_FP32_TFLOPS = None   # theoretical peak fp32 TFLOPS
PEAK_FP16_TFLOPS = None   # theoretical peak fp16 TFLOPS (tensor cores)
MEM_BW_TB_S      = None   # memory bandwidth in TB/s


# ---------------------------------------------------------------------------
# 2. Benchmark a single matmul
#    Returns: median elapsed time in seconds
# ---------------------------------------------------------------------------
def benchmark_matmul(N: int, dtype: torch.dtype, warmup: int = 5, iters: int = 20) -> float:
    """
    Run C = A @ B for square matrices of size N×N and return median time (s).

    Hints:
      - allocate A, B on CUDA with the given dtype
      - run a few warmup iterations before timing
      - use torch.cuda.Event(enable_timing=True) for accurate GPU timing
      - synchronize before reading elapsed time
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 3. Compute arithmetic intensity for square matmul
#    Returns: FLOP count, bytes moved, arithmetic intensity (FLOP/byte)
# ---------------------------------------------------------------------------
def matmul_arithmetic_intensity(N: int, dtype: torch.dtype) -> tuple[int, int, float]:
    """
    For C = A @ B with N×N matrices:
      - FLOP count: 2 * N^3  (N^3 multiplies + N^3 adds)
      - bytes moved: depends on dtype element size and number of matrices read/written

    Hints:
      - torch.finfo(dtype).bits // 8 gives bytes per element
      - count reads (A, B) and writes (C)
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. Sweep matrix sizes and collect measurements
# ---------------------------------------------------------------------------
def run_sweep(sizes: list[int], dtype: torch.dtype) -> dict:
    """
    For each N in sizes, benchmark matmul and compute:
      - measured TFLOPS
      - arithmetic intensity

    Returns a dict with lists: 'sizes', 'tflops', 'intensities'
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 5. Plot the roofline
# ---------------------------------------------------------------------------
def plot_roofline(fp32_results: dict, fp16_results: dict) -> None:
    """
    Draw the roofline model and overlay measured points.

    Steps:
      1. choose a range of arithmetic intensities (x-axis)
      2. compute the memory-bound ceiling:  min(BW * intensity, peak_compute)
         do this for both fp32 and fp16 ceilings
      3. plot the two roofline curves
      4. scatter-plot the measured (intensity, tflops) points for each dtype
      5. label axes, add legend, mark the ridge point
    """
    # TODO: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    sizes = [128, 256, 512, 1024, 2048, 4096, 8192]

    print("Benchmarking fp32...")
    fp32_results = run_sweep(sizes, torch.float32)

    print("Benchmarking fp16...")
    fp16_results = run_sweep(sizes, torch.float16)

    plot_roofline(fp32_results, fp16_results)
    plt.savefig("roofline.png", dpi=150)
    print("Saved roofline.png")
