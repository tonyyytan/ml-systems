"""
Reads roofline.csv produced by ./roofline and saves roofline.png.

Usage:
    python3 plot_roofline.py roofline.csv
"""

import sys
import csv
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Display style per series. cuBLAS series draw their roofline ceiling and use a
# filled circle; the naive custom-kernel series share the same ceiling (same
# peak/bandwidth) so we skip re-drawing it and use a hollow diamond marker.
SERIES = {
    "fp32":        {"color": "#4C72B0", "label": "FP32 (cuBLAS)",          "marker": "o", "ceiling": True},
    "fp16":        {"color": "#DD8452", "label": "FP16 cuBLAS (tensor)",   "marker": "o", "ceiling": True},
    "fp32_naive":  {"color": "#55A868", "label": "FP32 (naive kernel)",    "marker": "D", "ceiling": False},
    "fp16_naive":  {"color": "#C44E52", "label": "FP16 (naive kernel)",    "marker": "D", "ceiling": False},
}

def load_csv(path):
    ceiling = defaultdict(list)
    measured = defaultdict(list)

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dtype = row["dtype"]
            kind  = row["kind"]
            x     = float(row["intensity_or_x"])
            y     = float(row["tflops"])
            if kind == "ceiling":
                ceiling[dtype].append((x, y))
            else:
                size = int(row["size_or_x"])
                measured[dtype].append((size, x, y))

    return ceiling, measured

def plot(csv_path):
    ceiling, measured = load_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot any series present in the CSV; fall back to a default style for
    # unexpected labels so nothing is silently dropped.
    for dtype in sorted(set(ceiling) | set(measured)):
        style = SERIES.get(dtype, {"color": None, "label": dtype,
                                   "marker": "s", "ceiling": True})
        c = style["color"]

        # Roofline ceiling curve
        if style["ceiling"] and ceiling[dtype]:
            xs, ys = zip(*ceiling[dtype])
            line, = ax.plot(xs, ys, color=c, linewidth=2,
                            label=f"{style['label']} ceiling")
            c = line.get_color()

        # Measured points
        if measured[dtype]:
            _, intensities, tflops = zip(*measured[dtype])
            ax.scatter(intensities, tflops, color=c, zorder=5,
                       s=60, marker=style["marker"], edgecolors="white",
                       linewidths=0.8, label=f"{style['label']} measured")
            # Annotate each point with the matrix size N
            for n, xi, yi in measured[dtype]:
                ax.annotate(f"N={n}", (xi, yi),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=7, color=c)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))

    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance  (TFLOPS)", fontsize=12)
    ax.set_title("Roofline Model — RTX 5060 Laptop GPU (Legion 5)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    out = "roofline.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} roofline.csv")
        sys.exit(1)
    plot(sys.argv[1])
