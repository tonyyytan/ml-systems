"""
Project 03, Step 2: validate the two-tier roofline against llama.cpp.

Step 1 (roofline2.py) predicted decode tokens/sec as a function of how much of
the model spills to RAM. This is the reality check: take a model that does NOT
fit on the 8 GB card, sweep llama.cpp's -ngl (how many layers live on the GPU)
from 0 to all layers, measure tokens/sec at each split, and overlay the measured
points on the predicted curve from Step 1.

This is the gate for the whole project. If the measured points land on the line,
the model is real and everything downstream has a foundation. If they don't,
stop and find out why before writing a line of engine code.

The knob has to match, or the overlay isn't apples to apples. llama.cpp's -ngl
is the count of layers kept ON the gpu, which is the inverse of offload_frac:

    offload_frac = (num_layers - ngl) / num_layers

Caveat to check when the curve is read: -ngl also decides where each layer's KV
lives (gpu layer -> gpu KV), which is the same per-layer coupling roofline2
assumes, so the two should line up. Prompt-processing speed is a separate number;
we only compare the decode (token-generation) rate.
"""

import subprocess
import matplotlib.pyplot as plt
import json
import re

from roofline.roofline2 import LLAMA3_8B, predict_throughput, sweep_offload_fraction

# --- run config -------------------------------------------------------------
# TODO: fill these in on the machine. The GGUF must be a precision that spills
#       on 8 GB (fp16 ~16 GB, or Q8 ~8.5 GB) so there's actually a curve to walk.
LLAMA_CLI = None      # path to llama-cli / llama-bench binary
MODEL_GGUF = None     # path to the .gguf that does not fit in VRAM
BITS = 16             # precision of MODEL_GGUF, so the predicted curve matches
SEQ_LEN = 2048
BATCH = 1
N_PREDICT = 128       # decode tokens to time per run; enough to average out noise
DUMMY_PROMPT = "test " * int(SEQ_LEN * 0.75)


def ngl_to_offload_frac(ngl: int, num_layers: int) -> float:
    """-ngl (layers on gpu) -> offload_frac (fraction of layers in RAM)."""
    return max(0.0, (num_layers - ngl) / num_layers)


def run_llamacpp(ngl: int) -> float:
    """
    Run the model once at a given -ngl and return decode tokens/sec.

    Prefer llama-bench (its output already separates prompt-eval from token-gen
    and reports a clean tok/s); llama-cli works too but needs the timing lines
    parsed out of stderr. Either way, return ONLY the decode rate.
    """
    is_bench = "llama-bench" in LLAMA_CLI.lower()

    if is_bench:
        cmd = [LLAMA_CLI, "-m", MODEL_GGUF, "-ngl", str(ngl), "-p", str(SEQ_LEN), "-n", str(N_PREDICT), "-b", str(BATCH), "-o", "json"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            for entry in data:
                if entry.get("n_gen", 0) > 0:
                    return float(entry["avg_ts"])

            return 0.0  # no generation row found

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, IndexError, ValueError):
            return 0.0

    else:
        cmd = [LLAMA_CLI, "-m", MODEL_GGUF, "-ngl", str(ngl), "-p", DUMMY_PROMPT, "-n", str(N_PREDICT), "-c", str(SEQ_LEN), "--batch-size", str(BATCH), "--ignore-eos"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stderr
            # llama-cli prints two timing lines: "prompt eval time" (prefill) and
            # "eval time" (decode). We want decode only, so the lookbehind rejects
            # the "prompt " prefix and keeps the bare "eval time" line.
            match = re.search(r"(?<!prompt )eval time.*?\(\s*([\d.]+)\s+tokens per second\)", output)

            if match:
                return float(match.group(1))
            else:
                return 0.0


        except subprocess.CalledProcessError:
            return 0.0

def sweep_ngl(ngl_list: list[int]) -> dict:
    """
    Sweep -ngl and collect measured decode tok/s at each split. Returns lists
    keyed 'ngl', 'offload_frac', 'tps' so it can be overlaid straight onto the
    Step 1 prediction. Averaging repeats per point is worth it; decode rate on
    WSL2 is noisy.
    """

    result = {"ngl": [], "offload_frac": [], "tps": []}

    for ngl in ngl_list:
        print(f"Sweeping -ngl {ngl}...")
        runs = []

        for _ in range(3):
            tps = run_llamacpp(ngl)
            if tps > 0.0:
                runs.append(tps)
        
        avg_tps = sum(runs) / len(runs) if runs else 0.0

        result["ngl"].append(ngl)
        result["offload_frac"].append(ngl_to_offload_frac(ngl, LLAMA3_8B.num_layers))
        result["tps"].append(avg_tps)

    return result


# --- overlay ----------------------------------------------------------------

def plot_validation(measured: dict) -> None:
    """
    The verdict plot. The Step 1 predicted curve (predict_throughput swept over
    offload_frac) as a line, the llama.cpp measurements as points on top. If the
    points sit on the line the model holds; if they're a constant factor low,
    that factor is llama.cpp overhead worth naming, not a broken model.
    """
    model = LLAMA3_8B

    # Predicted line: the same Step 1 curve, swept fine (0..1) so it's smooth.
    fracs = [i / 100 for i in range(101)]
    predicted = sweep_offload_fraction(model, BITS, fracs, SEQ_LEN, BATCH)

    # Keep only points where a run actually produced a rate (drop the 0.0s from
    # failed/crashed runs so they don't drag the scatter to the floor).
    points = [(f, t, n) for f, t, n in
              zip(measured["offload_frac"], measured["tps"], measured["ngl"])
              if t > 0.0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Step 1 prediction as the reference line.
    ax.plot(predicted["offload_frac"], predicted["tps"],
            color="#DD8452", linewidth=2, zorder=1,
            label=f"predicted (roofline2, int{BITS})")

    # llama.cpp measurements on top.
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, color="#4C72B0", s=70, zorder=5,
                   edgecolors="white", linewidths=0.8,
                   label="measured (llama.cpp)")
        # Annotate each point with the -ngl it came from.
        for f, t, n in points:
            ax.annotate(f"ngl={n}", (f, t),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=8, color="#4C72B0")

    ax.set_xlabel("Offload fraction  (share of layers in RAM)", fontsize=12)
    ax.set_ylabel("Decode throughput  (tokens/sec)", fontsize=12)
    ax.set_title(f"Two-tier roofline validation — {model.name}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()


if __name__ == "__main__":
    assert None not in (LLAMA_CLI, MODEL_GGUF), \
        "fill LLAMA_CLI and MODEL_GGUF before running"

    model = LLAMA3_8B
    # -ngl 0 = all in RAM, -ngl num_layers = all on GPU. Include the endpoints.
    ngl_values = list(range(0, model.num_layers + 1, 4)) + [model.num_layers]

    measured = sweep_ngl(ngl_values)
    plot_validation(measured)
    plt.savefig("validate-llamacpp.png", dpi=150)
    print("Saved validate-llamacpp.png")
