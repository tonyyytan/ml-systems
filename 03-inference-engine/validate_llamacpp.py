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

from roofline2 import LLAMA3_8B, predict_throughput, sweep_offload_fraction

# --- run config -------------------------------------------------------------
# TODO: fill these in on the machine. The GGUF must be a precision that spills
#       on 8 GB (fp16 ~16 GB, or Q8 ~8.5 GB) so there's actually a curve to walk.
LLAMA_CLI = None      # path to llama-cli / llama-bench binary
MODEL_GGUF = None     # path to the .gguf that does not fit in VRAM
BITS = 16             # precision of MODEL_GGUF, so the predicted curve matches
SEQ_LEN = 2048
BATCH = 1
N_PREDICT = 128       # decode tokens to time per run; enough to average out noise


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
    # TODO: build the command (LLAMA_CLI, -m MODEL_GGUF, -ngl, -n N_PREDICT, a
    #       fixed prompt, -t threads), run it, parse the tok/s out of the output.
    #       Watch WSL2: confirm the run actually uses pinned H2D, since Step 0
    #       showed the pinned advantage is thin here.
    raise NotImplementedError


def sweep_ngl(ngl_list: list[int]) -> dict:
    """
    Sweep -ngl and collect measured decode tok/s at each split. Returns lists
    keyed 'ngl', 'offload_frac', 'tps' so it can be overlaid straight onto the
    Step 1 prediction. Averaging repeats per point is worth it; decode rate on
    WSL2 is noisy.
    """
    # TODO: loop ngl_list -> run_llamacpp, map ngl to offload_frac, collect.
    raise NotImplementedError


# --- overlay ----------------------------------------------------------------

def plot_validation(measured: dict) -> None:
    """
    The verdict plot. The Step 1 predicted curve (predict_throughput swept over
    offload_frac) as a line, the llama.cpp measurements as points on top. If the
    points sit on the line the model holds; if they're a constant factor low,
    that factor is llama.cpp overhead worth naming, not a broken model.
    """
    # TODO: plot sweep_offload_fraction(...) as the predicted line, scatter the
    #       measured points, annotate -ngl on each, match the roofline2 style.
    #       Save validate-llamacpp.png.
    raise NotImplementedError


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
