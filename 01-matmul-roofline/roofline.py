"""
Project A: Roofline plot for square fp32 and fp16 matmul (PyTorch version).
"""

import torch
import matplotlib.pyplot as plt

# TODO: fill in from the spec sheet / nvidia-smi
PEAK_FP32_TFLOPS = None
PEAK_FP16_TFLOPS = None   # tensor cores
MEM_BW_TB_S      = None


def benchmark_matmul(N: int, dtype: torch.dtype, warmup: int = 5, iters: int = 20) -> float:
    """Run C = A @ B for square N×N matrices, return median time (s)."""
    # TODO: implement
    raise NotImplementedError


def matmul_arithmetic_intensity(N: int, dtype: torch.dtype) -> tuple[int, int, float]:
    """Returns FLOP count (2 * N^3), bytes moved, arithmetic intensity (FLOP/byte)."""
    # TODO: implement
    raise NotImplementedError


def run_sweep(sizes: list[int], dtype: torch.dtype) -> dict:
    """Returns a dict with lists: 'sizes', 'tflops', 'intensities'."""
    # TODO: implement
    raise NotImplementedError


def plot_roofline(fp32_results: dict, fp16_results: dict) -> None:
    """Draw the roofline ceilings and overlay measured (intensity, tflops) points."""
    # TODO: implement
    raise NotImplementedError


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
