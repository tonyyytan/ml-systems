"""
Reads gemv_results.csv (from `make run`) and saves gemv-bandwidth.png.

Also times the pytorch baseline in-process, since the C++ harness only knows
about cuBLAS: `W @ x` is what someone would actually write, so it is the number
the kernels have to beat to justify existing. Needs the torch extension built
(`python3 setup.py build_ext --inplace`); without it the pytorch panel is
skipped and the bandwidth panel still plots.

Usage, from 03-inference-engine:
    python3 plot_gemv.py
"""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_IN = "gemv_results.csv"
PNG_OUT = "gemv-bandwidth.png"

PEAK_BW_GB_S = 384.0

WARMUP_ITERS = 20
TIMED_ITERS = 100
FLUSH_BYTES = 128 << 20

# categorical slots 1-3, validated all-pairs in both modes
C_SCALAR = "#2a78d6"
C_VEC = "#eb6834"
C_INT8 = "#1baf7a"
C_MUTED = "#8a8985"
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"

VARIANTS = [
    ("fp16_scalar", "fp16, scalar loads", C_SCALAR),
    ("fp16_vec", "fp16, float4 loads", C_VEC),
    ("w8a16", "w8a16, int8 + fused dequant", C_INT8),
]


def load_csv(path):
    rows = defaultdict(dict)
    ceiling = None
    order = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["variant"] == "stream_read":
                ceiling = float(row["gbs_min"])
                continue
            if row["shape"] not in order:
                order.append(row["shape"])
            rows[row["shape"]][row["variant"]] = row
    return order, rows, ceiling


def time_pytorch(order, rows):
    """Median ms for `W @ x` and for each kernel, same shapes, L2 flushed."""
    import torch
    import gemv_cuda

    from engine.quantize import quantize_w8a16

    torch.manual_seed(0)
    dev = torch.device("cuda")
    flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device=dev)

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    def time_op(fn):
        for _ in range(WARMUP_ITERS):
            fn()
        torch.cuda.synchronize()

        samples = []
        for _ in range(TIMED_ITERS):
            flush.zero_()
            start.record()
            fn()
            stop.record()
            stop.synchronize()
            samples.append(start.elapsed_time(stop))
        samples.sort()
        return samples[len(samples) // 2]

    out = {}
    for shape in order:
        M = int(rows[shape]["fp16_vec"]["M"])
        K = int(rows[shape]["fp16_vec"]["K"])

        W = torch.randn(M, K, dtype=torch.float16) * 0.05
        x = torch.randn(K, dtype=torch.float16) * 0.05
        q, s = quantize_w8a16(W)

        W_d, x_d, q_d, s_d = W.to(dev), x.to(dev), q.to(dev), s.to(dev)

        out[shape] = {
            "torch": time_op(lambda: W_d @ x_d),
            "fp16_scalar": time_op(lambda: gemv_cuda.gemv_fp16(W_d, x_d)),
            "fp16_vec": time_op(lambda: gemv_cuda.gemv_fp16_vec(W_d, x_d)),
            "w8a16": time_op(lambda: gemv_cuda.gemv_w8a16(q_d, s_d, x_d)),
        }
    return out


def bandwidth_panel(ax, order, rows, ceiling):
    width = 0.26
    xs = range(len(order))

    for i, (variant, label, color) in enumerate(VARIANTS):
        offset = (i - 1) * width
        vals = [float(rows[s][variant]["gbs_median"]) for s in order]
        bars = ax.bar([x + offset for x in xs], vals, width * 0.92, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7, color=C_TEXT_2)

    ax.axhline(ceiling, color=C_TEXT_2, linewidth=1, linestyle="--")
    ax.text(-0.48, ceiling + 7, f"measured ceiling {ceiling:.0f} GB/s",
            ha="left", fontsize=7.5, color=C_TEXT_2)

    ax.axhline(PEAK_BW_GB_S, color=C_MUTED, linewidth=1, linestyle=":")
    ax.text(-0.48, PEAK_BW_GB_S + 7, f"gddr7 spec peak {PEAK_BW_GB_S:.0f} GB/s",
            ha="left", fontsize=7.5, color=C_MUTED)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(order)
    ax.set_ylabel("achieved bandwidth (GB/s)")
    ax.set_ylim(0, PEAK_BW_GB_S * 1.22)
    ax.set_title("how close to the memory ceiling", fontsize=10.5, color=C_TEXT, loc="left", pad=10)


def speedup_panel(ax, order, torch_ms):
    width = 0.26
    xs = range(len(order))

    for i, (variant, label, color) in enumerate(VARIANTS):
        offset = (i - 1) * width
        vals = [torch_ms[s]["torch"] / torch_ms[s][variant] for s in order]
        bars = ax.bar([x + offset for x in xs], vals, width * 0.92, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}x",
                    ha="center", va="bottom", fontsize=7, color=C_TEXT_2)

    ax.axhline(1.0, color=C_TEXT_2, linewidth=1, linestyle="--")
    ax.text(len(order) - 0.5, 2.62, "dashed line = pytorch  W @ x", ha="right", fontsize=7.5, color=C_TEXT_2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(order)
    ax.set_ylabel("speedup over pytorch (wall clock)")
    ax.set_ylim(0, 2.75)
    ax.set_title("what that buys at batch 1", fontsize=10.5, color=C_TEXT, loc="left", pad=10)


def main():
    order, rows, ceiling = load_csv(CSV_IN)

    try:
        torch_ms = time_pytorch(order, rows)
    except Exception as e:
        print(f"pytorch panel skipped: {e}")
        torch_ms = None

    n = 2 if torch_ms else 1
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.0))
    axes = axes if n > 1 else [axes]

    bandwidth_panel(axes[0], order, rows, ceiling)
    if torch_ms:
        speedup_panel(axes[1], order, torch_ms)

    for ax in axes:
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(C_MUTED)
        ax.tick_params(colors=C_TEXT_2, length=0)

    axes[0].legend(frameon=False, fontsize=8, loc="lower right", ncol=1)

    fig.text(0.006, 0.965, "gemv at batch 1 is 1 flop per 2 bytes, so bytes read is the whole game",
             fontsize=13, color=C_TEXT, ha="left", va="top")
    fig.text(0.006, 0.917,
             "rtx 5060 laptop (gb206), llama-3-8b decode shapes, median of 100, l2 flushed between iterations. w8a16 bandwidth is\n"
             "computed on the bytes it actually reads (M*K + the scales), so the speedup panel is the apples-to-apples one.",
             fontsize=8, color=C_TEXT_2, ha="left", va="top", linespacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.savefig(PNG_OUT, dpi=160, facecolor="#fcfcfb")
    print(f"wrote {PNG_OUT}")


if __name__ == "__main__":
    main()
