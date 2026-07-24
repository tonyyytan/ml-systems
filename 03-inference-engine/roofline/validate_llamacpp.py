"""
Project 03, Step 2: validate the two-tier roofline against llama.cpp.

Sweep llama.cpp's -ngl (layers kept on the GPU) and measure decode tok/s at each
split, then overlay on the roofline2 prediction. -ngl is the inverse of offload:

    offload_frac = (num_layers - ngl) / num_layers

Two precisions so the overlay spans the offload axis: int4 (Q4_K_M) fits on 8 GB
and walks 0->100%, int8 (Q8_0) is forced to spill and covers the high end. Each
line is drawn at the GGUF's real on-disk bits/param, so a K-quant is compared
honestly. -d (n_depth) fills the KV cache to SEQ_LEN before timing decode, so the
measured KV traffic matches roofline2's KV term.
"""

import os
import sys
import subprocess
import json
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent

from roofline.roofline2 import (LLAMA3_8B, sweep_offload_fraction,
                                CPU_BW_GB_S, PCIE_BW_GB_S)

# --- run config -------------------------------------------------------------
LLAMA_BENCH = "/home/tanto/llama.cpp/build/bin/llama-bench"

MODELS = [
    {"label": "int4 (Q4_K_M)", "color": "#4C72B0",
     "path": "/home/tanto/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"},
    {"label": "int8 (Q8_0)", "color": "#55A868",
     "path": "/home/tanto/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"},
]

SEQ_LEN = 2048     # -d fills the KV cache to this depth so the measured KV term matches roofline2
BATCH = 1
N_PREDICT = 128
REPS = 3           # -r; llama-bench averages the repeats itself


def effective_bits(path: str) -> float:
    """Bits/param the file actually streams: on-disk bytes * 8 / params (Q4_K_M ~4.9)."""
    return os.path.getsize(path) * 8 / LLAMA3_8B.num_params


def ngl_to_offload_frac(ngl: int, num_layers: int) -> float:
    """-ngl (layers on gpu) -> offload_frac (fraction of layers in RAM)."""
    return max(0.0, (num_layers - ngl) / num_layers)


def run_llamacpp(ngl: int, model_path: str) -> float:
    """
    Run llama-bench once at a given -ngl and return decode tok/s (the n_gen > 0
    row's avg_ts). A CUDA OOM exits nonzero and is caught as 0.0, which the plot drops.
    """
    cmd = [LLAMA_BENCH, "-m", model_path, "-ngl", str(ngl),
           "-p", "0", "-n", str(N_PREDICT), "-d", str(SEQ_LEN),
           "-r", str(REPS), "-o", "json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for entry in data:
            if entry.get("n_gen", 0) > 0:
                return float(entry["avg_ts"])
        return 0.0
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
        return 0.0


def sweep_ngl(ngl_list: list[int], model_path: str) -> dict:
    """Sweep -ngl, collect decode tok/s at each split. llama-bench averages REPS itself."""
    result = {"ngl": [], "offload_frac": [], "tps": []}
    for ngl in ngl_list:
        tps = run_llamacpp(ngl, model_path)
        print(f"  -ngl {ngl:>2}: {tps:6.2f} tok/s" if tps else f"  -ngl {ngl:>2}: OOM/skip")
        result["ngl"].append(ngl)
        result["offload_frac"].append(ngl_to_offload_frac(ngl, LLAMA3_8B.num_layers))
        result["tps"].append(tps)
    return result


# --- overlay ----------------------------------------------------------------

def plot_validation(runs: list[dict]) -> None:
    """
    Per precision, two predicted lines from the same equation, different offload
    slopes: CPU-offload (solid, what llama.cpp -ngl does, points should land here)
    and PCIe-stream (dashed, this engine's naive floor, shown for contrast).
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fracs = [i / 100 for i in range(101)]

    for r in runs:
        cpu_line = sweep_offload_fraction(LLAMA3_8B, r["bits"], fracs, SEQ_LEN, BATCH,
                                          offload_bw_gb_s=CPU_BW_GB_S)
        pcie_line = sweep_offload_fraction(LLAMA3_8B, r["bits"], fracs, SEQ_LEN, BATCH,
                                           offload_bw_gb_s=PCIE_BW_GB_S)
        ax.plot(cpu_line["offload_frac"], cpu_line["tps"],
                color=r["color"], linewidth=2, alpha=0.7, zorder=2,
                label=f"{r['label']} predicted — CPU-offload {CPU_BW_GB_S} GB/s")
        ax.plot(pcie_line["offload_frac"], pcie_line["tps"],
                color=r["color"], linewidth=1.3, alpha=0.4, linestyle="--", zorder=1,
                label=f"{r['label']} predicted — PCIe-stream {PCIE_BW_GB_S} GB/s (engine floor)")

        # More GPU layers can only be faster; a point slower than one with fewer
        # offloaded layers hit VRAM oversubscription (Q8 at ngl=32 on 8 GB). Drop
        # those and the 0.0s. Swept order is ngl ascending, so track a running max.
        points, best = [], 0.0
        for f, t, n in zip(r["measured"]["offload_frac"], r["measured"]["tps"],
                           r["measured"]["ngl"]):
            if t <= 0.0 or t < best:
                continue
            best = t
            points.append((f, t, n))
        if points:
            ax.scatter([p[0] for p in points], [p[1] for p in points],
                       color=r["color"], s=70, zorder=5,
                       edgecolors="white", linewidths=0.8,
                       label=f"{r['label']} measured (llama.cpp -ngl)")
            for f, t, n in points:
                ax.annotate(f"ngl={n}", (f, t),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=7, color=r["color"])

    ax.set_xlabel("Offload fraction  (share of layers out of VRAM)", fontsize=12)
    ax.set_ylabel("Decode throughput  (tokens/sec)", fontsize=12)
    ax.set_title(f"Two-tier roofline validation — {LLAMA3_8B.name}, "
                 f"seq {SEQ_LEN}, batch {BATCH}", fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()


CACHE = OUT_DIR / "sweep_results.json"

if __name__ == "__main__":
    # Re-plot from cached measurements unless --resweep is passed (the sweep is slow).
    if CACHE.exists() and "--resweep" not in sys.argv:
        runs = json.load(open(CACHE))
    else:
        assert os.path.exists(LLAMA_BENCH), f"build llama-bench first: {LLAMA_BENCH}"
        ngl_values = list(range(0, LLAMA3_8B.num_layers + 1, 4))
        if LLAMA3_8B.num_layers not in ngl_values:
            ngl_values.append(LLAMA3_8B.num_layers)

        runs = []
        for m in MODELS:
            assert os.path.exists(m["path"]), f"missing GGUF: {m['path']}"
            bits = effective_bits(m["path"])
            print(f"Sweeping {m['label']} — {bits:.2f} bits/param")
            measured = sweep_ngl(ngl_values, m["path"])
            runs.append({**m, "bits": bits, "measured": measured})
        json.dump(runs, open(CACHE, "w"), indent=2)

    plot_validation(runs)
    plt.savefig(OUT_DIR / "validate-llamacpp.png", dpi=150)
    print(f"Saved {OUT_DIR / 'validate-llamacpp.png'}")
