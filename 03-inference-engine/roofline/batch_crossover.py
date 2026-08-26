"""
Project 03, Step 2b: the batch axis. Where does streaming overtake CPU offload?

Step 2 validated the roofline at batch 1 and stopped there, which is the one
batch where the answer is uninteresting: every tier is bandwidth bound, 48 GB/s
beats 14 GB/s, llama.cpp wins by 3.5x. The tiers only separate once compute
starts binding, and they bind at very different batches because the processors
are 69x apart in flops.

    cpu-offload : reads weights out of DDR5 once, then the CPU does b tokens of
                  math with them. bandwidth bound at b=1, compute bound by b~4,
                  and flat forever after -- the ceiling is CPU_TFLOPS.
    pcie-stream : moves the same weight bytes regardless of b, and the GPU is
                  fast enough that the copy never stops being the binding term
                  at any batch worth running. throughput climbs linearly.

One curve flattens, one keeps climbing, so they cross. b* is that crossing, and
it is the number that says whether this engine has a window to win in.

For context on what to beat: llama.cpp does have a streaming path, but it fires
on a constant a human picks (ggml-cuda.cu, op_offload_min_batch_size, 32 by
default, GGML_OP_OFFLOAD_MIN_BATCH overrides) rather
than on anything measured about the machine.

    python3 -m roofline.batch_crossover
"""

from pathlib import Path
import matplotlib.pyplot as plt

from roofline.roofline2 import (LLAMA3_8B, TIER_CPU, TIER_STREAM, TIER_STREAM_NAIVE,
                                CPU_TFLOPS, GPU_TFLOPS, PCIE_BW_GB_S, CPU_BW_GB_S,
                                sweep_batch, crossover_batch, compute_bound_batch,
                                predict_throughput, weight_bytes)

OUT_DIR = Path(__file__).parent

BITS = 4.9        # Q4_K_M on disk, the same effective bits step 2 validated at
SEQ_LEN = 2048
OFFLOAD_FRAC = 1.0
LLAMACPP_MIN_BATCH = 32   # ggml-cuda.cu:5507 default, the constant this is all measured against

BATCHES = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512]

# Measured today on Q4_K_M, all at -ngl 0 unless noted. The first two set
# CPU_BW and CPU_TFLOPS, so they are fits. The third is out of sample: the model
# predicts it from those plus the 14 GB/s PCIe slope from step 0.
#
# The pp rows carry their own seq_len: prompt processing starts from an empty
# cache, so its KV traffic is a fraction of decode-at-depth-2048. Halfway through
# the prompt is the average depth. That makes the KV term approximate for those
# two rows (and attention flops are ignored throughout), which is fine here --
# at batch 512 the weights and the matmul dominate either way.
MEASURED = [
    ("decode b=1, CPU offload", 1, 2048, TIER_CPU, 9.23, "fit (sets CPU_BW)"),
    ("pp512, -nopo 1 (pure CPU)", 512, 256, TIER_CPU, 37.40, "fit (sets CPU_TFLOPS)"),
    ("pp512, op offload (streamed)", 512, 256, TIER_STREAM_NAIVE, 855.42, "predicted"),
]


def validation_table() -> None:
    print(f"\nmodel vs measured  ({LLAMA3_8B.name}, {BITS} bits/param, 100% offload)")
    print(f"{'run':<30} {'measured':>10} {'predicted':>10} {'err':>7}   note")
    for label, batch, seq_len, tier, measured, note in MEASURED:
        pred = predict_throughput(LLAMA3_8B, BITS, OFFLOAD_FRAC, seq_len, batch, tier)
        err = (pred - measured) / measured * 100
        print(f"{label:<30} {measured:>10.1f} {pred:>10.1f} {err:>+6.1f}%   {note}")


def crossover_report() -> float | None:
    b_star = crossover_batch(LLAMA3_8B, BITS, SEQ_LEN, OFFLOAD_FRAC, TIER_CPU, TIER_STREAM)
    b_cpu = compute_bound_batch(LLAMA3_8B, BITS, TIER_CPU)
    b_gpu = compute_bound_batch(LLAMA3_8B, BITS, TIER_STREAM)

    weights_gb = weight_bytes(LLAMA3_8B, BITS) / 1e9
    print(f"\nweights {weights_gb:.2f} GB, ceilings {CPU_TFLOPS} / {GPU_TFLOPS} TFLOP/s, "
          f"slopes {CPU_BW_GB_S} / {PCIE_BW_GB_S} GB/s")
    print(f"  cpu-offload goes compute bound at b = {b_cpu:.1f}")
    print(f"  pcie-stream goes compute bound at b = {b_gpu:.0f}  (never, in practice)")

    if b_star is None:
        print("  no crossover: cpu-offload wins at every batch")
        return None

    tps_cpu = predict_throughput(LLAMA3_8B, BITS, OFFLOAD_FRAC, SEQ_LEN, int(b_star), TIER_CPU)
    tps_str = predict_throughput(LLAMA3_8B, BITS, OFFLOAD_FRAC, SEQ_LEN, int(b_star), TIER_STREAM)
    print(f"  b* = {b_star:.0f}   (cpu {tps_cpu:.1f} vs stream {tps_str:.1f} tok/s)")

    if b_star < LLAMACPP_MIN_BATCH:
        lo, hi = int(b_star), LLAMACPP_MIN_BATCH - 1
        gain = (predict_throughput(LLAMA3_8B, BITS, OFFLOAD_FRAC, SEQ_LEN, hi, TIER_STREAM) /
                predict_throughput(LLAMA3_8B, BITS, OFFLOAD_FRAC, SEQ_LEN, hi, TIER_CPU))
        print(f"  window: batch {lo}-{hi} llama.cpp stays on the CPU tier "
              f"(its threshold is {LLAMACPP_MIN_BATCH}); at b={hi} that costs {gain:.1f}x")
    return b_star


def plot_crossover(b_star: float | None) -> None:
    cpu = sweep_batch(LLAMA3_8B, BITS, BATCHES, SEQ_LEN, OFFLOAD_FRAC, TIER_CPU)
    stream = sweep_batch(LLAMA3_8B, BITS, BATCHES, SEQ_LEN, OFFLOAD_FRAC, TIER_STREAM)
    naive = sweep_batch(LLAMA3_8B, BITS, BATCHES, SEQ_LEN, OFFLOAD_FRAC, TIER_STREAM_NAIVE)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(cpu["batch"], cpu["tps"], color="#4C72B0", linewidth=2, marker="o", markersize=4,
            label=f"cpu-offload — {CPU_BW_GB_S} GB/s, {CPU_TFLOPS} TFLOP/s (llama.cpp -ngl)")
    ax.plot(stream["batch"], stream["tps"], color="#DD8452", linewidth=2, marker="o", markersize=4,
            label=f"pcie-stream, prefetched — {PCIE_BW_GB_S} GB/s, {GPU_TFLOPS} TFLOP/s (this engine)")
    ax.plot(naive["batch"], naive["tps"], color="#DD8452", linewidth=1.3, linestyle="--", alpha=0.6,
            label="pcie-stream, copy then compute (llama.cpp op offload)")

    for label, batch, seq_len, tier, measured, note in MEASURED:
        ax.scatter([batch], [measured], s=90, zorder=5, color="#C44E52",
                   edgecolors="white", linewidths=0.8)
        ax.annotate(f"{label}\n{measured:.1f} tok/s ({note})", (batch, measured),
                    textcoords="offset points", xytext=(-10, 10), fontsize=7,
                    color="#C44E52", ha="right")

    if b_star:
        ax.axvline(b_star, color="#555555", linestyle=":", linewidth=1.5)
        ax.annotate(f"b* = {b_star:.0f}", (b_star, ax.get_ylim()[1] * 0.5),
                    textcoords="offset points", xytext=(6, 0), fontsize=10)
    ax.axvspan(b_star or 1, LLAMACPP_MIN_BATCH, color="#DD8452", alpha=0.08)
    ax.axvline(LLAMACPP_MIN_BATCH, color="#937860", linestyle="-.", linewidth=1.3,
               label=f"llama.cpp op-offload threshold ({LLAMACPP_MIN_BATCH}, its default)")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size  (tokens per decode step)", fontsize=12)
    ax.set_ylabel("Decode throughput  (tokens/sec)", fontsize=12)
    ax.set_title(f"Where streaming overtakes CPU offload — {LLAMA3_8B.name}, "
                 f"{BITS} bits, 100% offloaded, seq {SEQ_LEN}", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.5, -0.13, "lines are decode at seq 2048; the two pp512 points are prefill from an "
                        "empty cache, so they sit above the lines by their missing KV traffic",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    fig.tight_layout()


if __name__ == "__main__":
    validation_table()
    b_star = crossover_report()
    plot_crossover(b_star)
    plt.savefig(OUT_DIR / "batch-crossover.png", dpi=150)
    print(f"\nSaved {OUT_DIR / 'batch-crossover.png'}")
