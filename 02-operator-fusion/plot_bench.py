"""
Reads results.csv produced by benchmark.py and saves bench_benchmark.png.

Usage:
    python3 plot_bench.py results.csv

CSV format:
    op,variant,N,time_ms,bw_gb_s
"""

import sys
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style per variant
VARIANT_STYLE = {
    "eager":   {"linestyle": ":",  "marker": "s"},
    "compile": {"linestyle": "--", "marker": "o"},
    "custom":  {"linestyle": "-",  "marker": "^"},
}

# Color per op
OP_COLOR = {
    "relu":          "#4C72B0",
    "bias_gelu":     "#DD8452",
    "add_layernorm": "#55A868",
}

def load_csv(path):
    # data[op][variant] = [(N, bw)]
    data = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op      = row["op"]
            variant = row["variant"]
            N       = int(row["N"])
            bw      = float(row["bw_gb_s"])
            data[op][variant].append((N, bw))
    return data

def plot(csv_path):
    data = load_csv(csv_path)

    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5), sharey=False)
    if len(data) == 1:
        axes = [axes]

    # GB/s RTX 5060 laptop
    peak_bw = 384.0

    for ax, (op, variants) in zip(axes, data.items()):
        color = OP_COLOR.get(op, None)
        ax.axhline(peak_bw, color="red", linewidth=1, linestyle="--", label="Peak BW")

        for variant, points in sorted(variants.items()):
            points.sort()
            Ns, bws = zip(*points)
            style = VARIANT_STYLE.get(variant, {"linestyle": "-", "marker": "x"})
            ax.plot(Ns, bws, color=color, label=variant, linestyle=style["linestyle"], marker=style["marker"], linewidth=1.5, markersize=6)

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel("N (elements)", fontsize=11)
        ax.set_ylabel("Effective BW  (GB/s)", fontsize=11)
        ax.set_title(op.replace("_", " ").title(), fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    fig.suptitle("Eager vs torch.compile vs Custom Kernel — RTX 5060 Laptop GPU", fontsize=13)
    fig.tight_layout()
    out = "bench_benchmark.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} results.csv")
        sys.exit(1)
    plot(sys.argv[1])
