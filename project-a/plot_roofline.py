"""
Reads roofline.csv produced by ./roofline and saves roofline.png.

Usage:
    python3 plot_roofline.py roofline.csv
"""

import sys
import csv
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_csv(path):
    ceiling = {"fp32": [], "fp16": []}
    measured = {"fp32": [], "fp16": []}

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

    colors = {"fp32": "#4C72B0", "fp16": "#DD8452"}
    labels = {"fp32": "FP32", "fp16": "FP16 (tensor cores)"}

    for dtype in ("fp32", "fp16"):
        c = colors[dtype]

        # Roofline ceiling curve
        if ceiling[dtype]:
            xs, ys = zip(*ceiling[dtype])
            ax.plot(xs, ys, color=c, linewidth=2, label=f"{labels[dtype]} ceiling")

        # Measured points
        if measured[dtype]:
            _, intensities, tflops = zip(*measured[dtype])
            sizes = [m[0] for m in measured[dtype]]
            sc = ax.scatter(intensities, tflops, color=c, zorder=5,
                            s=60, marker="o", edgecolors="white", linewidths=0.8)
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
