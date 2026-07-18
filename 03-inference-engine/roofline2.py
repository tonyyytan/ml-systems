"""
Project 03, Step 1: the two-tier roofline. THE deliverable.

01 asked "compute or bandwidth bound?" for a resident workload. This asks the
same question once the model no longer fits: some layers live in VRAM (streamed
at ~272 GB/s) and the rest spill to system RAM (streamed across PCIe at ~25
GB/s). Decode has to read every weight to make one token, so per-token time is
the sum of two bandwidth terms, not one:

    t_token  ~=  bytes_in_vram / VRAM_BW  +  bytes_in_ram / PCIE_BW

That second slope is the whole project. This file predicts decode throughput as
a function of model size, precision, and the fraction of layers offloaded, and
draws the cliff. Everything else in 03 exists to test the curve this file draws.

Nothing here needs the engine. It's arithmetic plus the two bandwidth numbers
measured in Step 0.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt

# --- hardware constants -----------------------------------------------------
# VRAM_BW is the resident slope (reuse the 01 measurement). PCIE_BW is the
# offload slope and is the second slope of the entire project: it MUST come from
# the Step 0 cudaMemcpy microbenchmark (pinned H2D), not a spec sheet, and on
# WSL2 the pinned-vs-pageable gap decides whether prefetch is even viable.
# Filled from Step 0 on the Legion 5 / RTX 5060 (Blackwell GB206), WSL2.
VRAM_BW_GB_S = 272    # resident slope, GB206 peak (reused from 01/02)
PCIE_BW_GB_S = 14     # offload slope, MEASURED pinned H2D on WSL2 (pageable ~13,
                      # so the pinned gap is thin here). below the ~16-32 hoped for.
VRAM_CAPACITY_GB = 8  # 5060 laptop has 8 GB (8151 MiB). sets where the cliff sits.

# The KV cache stays fp16 even when the weights are quantized to int8/int4, so
# it gets its own precision independent of the `bits` weights are streamed at.
KV_BITS = 16


@dataclass
class ModelConfig:
    """Enough of a model to size its weight and KV-cache traffic per token."""
    name: str
    num_layers: int
    hidden_dim: int
    num_params: int
    num_attention_heads: int
    num_kv_heads: int


# Reference targets. llama-3-8b is the one that makes the cliff sweepable:
# fp16 ~16 GB spills, int8 ~8 GB on the knife edge, int4 ~4 GB fits with KV room.
LLAMA3_8B = ModelConfig(name="llama-3-8b", num_layers=32, hidden_dim=4096, num_params=8_030_000_000, num_attention_heads = 32, num_kv_heads = 8)


def bytes_per_param(bits: int) -> float:
    """Bytes each weight crosses the bus as, at a given precision (16/8/4)."""
    return bits / 8


def weight_bytes(model: ModelConfig, bits: int) -> float:
    """Total weight bytes streamed to produce one token (decode reads all of them)."""
    return model.num_params * bytes_per_param(bits)


def kv_bytes_per_token(model: ModelConfig, seq_len: int, batch: int) -> float:
    """
    KV-cache bytes read per decode step, and equally the bytes it occupies in
    memory. Grows with seq_len and batch. Held at KV_BITS, not the weight bits.
    """
    head_dim = model.hidden_dim / model.num_attention_heads
    return 2 * head_dim * model.num_kv_heads * model.num_layers * seq_len * batch * bytes_per_param(KV_BITS)


def footprint_bytes(model: ModelConfig, bits: int, seq_len: int, batch: int) -> float:
    """What a fully resident copy costs in VRAM: weights + the whole KV cache.
    This is the number the VRAM budget has to clear, so it's what offload is
    measured against."""
    return weight_bytes(model, bits) + kv_bytes_per_token(model, seq_len, batch)


def decode_time_per_token(model: ModelConfig, bits: int, offload_frac: float, seq_len: int, batch: int) -> float:
    """
    The two-tier equation. offload_frac in [0,1] is the fraction of layers that
    spill to RAM; those bytes pay the PCIE slope, the rest pay the VRAM slope.

        t = resident_bytes / VRAM_BW  +  offloaded_bytes / PCIE_BW
    """
    total_weights = weight_bytes(model, bits)
    total_kv = kv_bytes_per_token(model, seq_len, batch)

    ram_bytes = (total_weights * offload_frac) + (total_kv * offload_frac)
    vram_bytes = (total_weights * (1 - offload_frac)) + (total_kv * (1 - offload_frac))

    time = vram_bytes / (VRAM_BW_GB_S * 1e9) + ram_bytes / (PCIE_BW_GB_S * 1e9)
    return time

def predict_throughput(model: ModelConfig, bits: int, offload_frac: float, seq_len: int, batch: int) -> float:
    """Predicted decode throughput in tokens/sec = batch / t_token."""
    # batch / (s / token * batch) = token / s 
    return batch / decode_time_per_token(model, bits, offload_frac, seq_len, batch)


def offload_fraction(model: ModelConfig, bits: int, vram_gb: float, seq_len: int, batch: int) -> float:
    """
    How much of the model spills given a VRAM budget. This is where the cliff
    lives: at the precision that just fits, offload_frac drops to 0 and the
    PCIE term vanishes from decode_time_per_token.

    Measured against the full footprint (weights + KV), because a resident layer
    has to fit both in VRAM. Ignoring KV here under-offloads and overpredicts tps.
    """
    footprint_gb = footprint_bytes(model, bits, seq_len, batch) / 1e9
    return min(1.0, max(0.0, (footprint_gb - vram_gb) / footprint_gb))

def arithmetic_intensity(model: ModelConfig, bits: int, seq_len: int, batch: int) -> float:
    """
    FLOP per byte at a given batch. At batch 1 this is ~1-2 (memory bound, the
    wall). The point of the sweep is finding the batch where compute finally
    hides a PCIE transfer, i.e. where prefetch stops being free money lost.
    """
    # 2 flops per bytes transferred
    return 2 * model.num_params * batch / (weight_bytes(model, bits) + kv_bytes_per_token(model, seq_len, batch))

# --- sweeps -----------------------------------------------------------------

def sweep_precision(model: ModelConfig, bits_list: list[int], vram_gb: float, seq_len: int, batch: int) -> dict:
    """
    The cliff sweep. For each precision: does it fit, what's offload_frac, and
    the predicted tokens/sec. Returns lists keyed 'bits', 'offload_frac',
    'tokens_per_sec', 'model_gb'.
    """

    results = {"bits": [], "offload_frac": [], "tps": [], "model_gb": []}

    for bits in bits_list:
        model_gb = weight_bytes(model, bits) / 1e9
        offload_frac = offload_fraction(model, bits, vram_gb, seq_len, batch)
        tps = predict_throughput(model, bits, offload_frac, seq_len, batch)
        results["bits"].append(bits)
        results["offload_frac"].append(offload_frac)
        results["tps"].append(tps)
        results["model_gb"].append(model_gb)

    return results


def sweep_offload_fraction(model: ModelConfig, bits: int, fracs_list: list[float], seq_len: int, batch: int) -> dict:
    """
    The continuous version: force offload_frac from 0 to 1 and predict
    tokens/sec at each point. THIS is the curve Step 2 overlays llama.cpp
    -ngl measurements onto. Returns 'offload_frac', 'tokens_per_sec'.
    """
    
    results = {"offload_frac": [], "tps": []}

    for frac in fracs_list:
        tps = predict_throughput(model, bits, frac, seq_len, batch)
        results["offload_frac"].append(frac)
        results["tps"].append(tps)

    return results

# --- plot -------------------------------------------------------------------

def plot_two_tier(precision_sweep: dict, offload_sweep: dict) -> None:
    """
    The artifact. Two panels:
      (a) tokens/sec vs precision, with the VRAM boundary drawn as a vertical
          line so the step at the fit/no-fit crossing is visible as a cliff.
      (b) tokens/sec vs offload fraction, the line Step 2 validates against.
    Mark VRAM_CAPACITY_GB and annotate where each precision lands.
    """
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) The cliff. x is the model's weight footprint so the VRAM boundary is a
    # single vertical line: points left of it fit and run at the VRAM slope,
    # points right of it spill and collapse onto the PCIE slope.
    model_gb = precision_sweep["model_gb"]
    tps = precision_sweep["tps"]
    ax_a.plot(model_gb, tps, color="#4C72B0", linewidth=1.5, zorder=1)
    ax_a.scatter(model_gb, tps, color="#4C72B0", s=70, zorder=5,
                 edgecolors="white", linewidths=0.8)
    for gb, y, bits, frac in zip(model_gb, tps, precision_sweep["bits"],
                                 precision_sweep["offload_frac"]):
        ax_a.annotate(f"int{bits}\noff={frac:.0%}", (gb, y),
                      textcoords="offset points", xytext=(6, 6),
                      fontsize=8, color="#4C72B0")

    ax_a.axvline(VRAM_CAPACITY_GB, color="#C44E52", linestyle="--", linewidth=1.5,
                 label=f"VRAM capacity ({VRAM_CAPACITY_GB} GB)")
    ax_a.set_xlabel("Model weight footprint  (GB)", fontsize=12)
    ax_a.set_ylabel("Predicted decode throughput  (tokens/sec)", fontsize=12)
    ax_a.set_title("Precision cliff", fontsize=13)
    ax_a.legend(fontsize=10)
    ax_a.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    # (b) The continuous curve Step 2 overlays -ngl measurements onto.
    ax_b.plot(offload_sweep["offload_frac"], offload_sweep["tps"],
              color="#DD8452", linewidth=2, marker="o", markersize=4,
              markeredgecolor="white", markeredgewidth=0.6)
    ax_b.set_xlabel("Offload fraction  (share of layers in RAM)", fontsize=12)
    ax_b.set_ylabel("Predicted decode throughput  (tokens/sec)", fontsize=12)
    ax_b.set_title("Throughput vs offload fraction", fontsize=13)
    ax_b.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    fig.suptitle(f"Two-tier roofline — {LLAMA3_8B.name}, "
                 f"{VRAM_BW_GB_S}/{PCIE_BW_GB_S} GB/s slopes", fontsize=14)
    fig.tight_layout()


if __name__ == "__main__":
    assert None not in (VRAM_BW_GB_S, PCIE_BW_GB_S, VRAM_CAPACITY_GB), \
        "fill hardware constants from Step 0 before running"

    model = LLAMA3_8B
    seq_len = 2048
    batch = 1

    print(f"Model: {model.name}, VRAM budget {VRAM_CAPACITY_GB} GB, "
          f"slopes {VRAM_BW_GB_S}/{PCIE_BW_GB_S} GB/s")

    prec = sweep_precision(model, [16, 8, 4], VRAM_CAPACITY_GB, seq_len, batch)
    off  = sweep_offload_fraction(model, 16, [i / 10 for i in range(11)],
                                  seq_len, batch)

    plot_two_tier(prec, off)
    plt.savefig("two-tier-roofline.png", dpi=150)
    print("Saved two-tier-roofline.png")
