"""
Project 03, Step 1: the two-tier roofline. THE deliverable.

01 asked "compute or bandwidth bound?" for a resident workload. This asks the
same question once the model no longer fits: some layers live in VRAM (streamed
at ~272 GB/s achieved, 71% of the card's 384 peak) and the rest spill to system
RAM (streamed across PCIe at ~14 GB/s, measured, not spec). Decode has to read
every weight to make one token, so per-token time is
the sum of two bandwidth terms, not one:

    t_token  ~=  bytes_in_vram / VRAM_BW  +  bytes_in_ram / PCIE_BW

That second slope is the whole project. This file predicts decode throughput as
a function of model size, precision, and the fraction of layers offloaded, and
draws the cliff. Everything else in 03 exists to test the curve this file draws.

Nothing here needs the engine. It's arithmetic plus the two bandwidth numbers
measured in Step 0.
"""

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent

# --- hardware constants -----------------------------------------------------
# Measured on the Legion 5 / RTX 5060 (Blackwell GB206), WSL2. PCIE_BW is the
# offload slope from the Step 0 pinned-H2D microbenchmark, not a spec sheet.
# The resident slope is an ACHIEVED rate, not the spec peak -- nothing streams at
# peak. Kept as peak x efficiency so the assumption is visible and swappable:
# STREAM_EFF is llama.cpp's, which is what step 2 validates against. Step 4's own
# gemv already sustains 310-345 GB/s (0.81-0.90), so the engine's resident tier
# should beat this once the kernels are wired in -- that headroom is the point.
PEAK_BW_GB_S = 384    # GB206 spec: 128-bit GDDR7 @ 24 Gbps. ceiling, never reached.
STREAM_EFF = 0.707    # MEASURED: llama.cpp Q4_K_M, ngl=32, seq 2048 -> 52.35 tok/s
                      # x 5.19 GB/token = 271.5 GB/s. see validate_llamacpp.py.
VRAM_BW_GB_S = round(PEAK_BW_GB_S * STREAM_EFF, 1)   # 271.5, the slope step 2 validates
PCIE_BW_GB_S = 14     # offload slope, MEASURED pinned H2D on WSL2 (pageable ~13,
                      # so the pinned gap is thin here). below the ~16-32 hoped for.
CPU_BW_GB_S = 48      # second offload tier. llama.cpp -ngl computes offloaded layers
                      # on the CPU from system RAM (DDR5), not over PCIe. measured from
                      # the ngl=0 endpoint. see validate_llamacpp.py.
VRAM_CAPACITY_GB = 8  # 5060 laptop has 8 GB (8151 MiB). sets where the cliff sits.

# Compute ceilings. Batch 1 decode never touches these (1-2 flop/byte keeps every
# tier bandwidth bound), which is why steps 1-2 could ignore them. Past batch ~4
# the CPU tier goes compute bound and the bandwidth-only model stops predicting.
# Both measured with llama-bench on Q4_K_M, effective flops = 2 * params * pp_tps:
#   CPU: -ngl 0 -nopo 1 -p 512 -> 37.4 tok/s  (op offload disabled, pure CPU)
#   GPU: -ngl 99      -p 512 -> 2592 tok/s   (fully resident)
CPU_TFLOPS = 0.60
GPU_TFLOPS = 41.6

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


@dataclass
class Tier:
    """
    Where an offloaded layer runs. A tier is a bandwidth AND a compute ceiling,
    because which one binds flips with batch size — that flip is the whole point
    of the batch sweep.

    overlap says whether the weight movement hides under the math. True is a
    prefetched copy stream (t = max), False is copy-then-compute (t = sum), which
    is what llama.cpp's op offload does today: the weight is an input to the
    graph split, recopied every eval behind a synchronize.

    kv_follows_weights says whether an offloaded layer's KV cache goes with it.
    It does under -ngl (the CPU computes that layer, so it reads its own KV out
    of system RAM) and it does not under streaming (the weights come to the GPU,
    the cache stays put).
    """
    name: str
    bw_gb_s: float
    tflops: float
    overlap: bool = True
    kv_follows_weights: bool = False


TIER_RESIDENT = Tier("resident", VRAM_BW_GB_S, GPU_TFLOPS)
TIER_CPU = Tier("cpu-offload", CPU_BW_GB_S, CPU_TFLOPS, kv_follows_weights=True)
TIER_STREAM = Tier("pcie-stream", PCIE_BW_GB_S, GPU_TFLOPS)
TIER_STREAM_NAIVE = Tier("pcie-stream (no prefetch)", PCIE_BW_GB_S, GPU_TFLOPS, overlap=False)


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


def flops_per_step(model: ModelConfig, batch: int) -> float:
    """
    Arithmetic in one decode step: 2 flop per weight per token in the batch. The
    term that was missing from steps 1-2. Flat in bytes, linear in batch, which
    is why batching is the knob that changes which ceiling binds.
    """
    return 2 * model.num_params * batch


def tier_time(bytes_moved: float, flops: float, tier: Tier) -> float:
    """
    Time for one tier's share of a step. Bandwidth and compute are two ceilings
    over the same work: with a prefetch stream the slower one sets the pace, and
    without one they serialise.
    """
    t_bw = bytes_moved / (tier.bw_gb_s * 1e9)
    t_compute = flops / (tier.tflops * 1e12)
    return max(t_bw, t_compute) if tier.overlap else t_bw + t_compute


def decode_time_per_token(model: ModelConfig, bits: int, offload_frac: float, seq_len: int, batch: int,
                          offload_tier: Tier = TIER_STREAM) -> float:
    """
    The two-tier equation, now with a compute term on each tier:

        t = tier_time(resident bytes, resident flops, RESIDENT)
          + tier_time(offloaded bytes, offloaded flops, offload_tier)

    At batch 1 both tiers are bandwidth bound and this reduces exactly to the
    bandwidth-only sum steps 1-2 validated to +/-6%. Past batch ~4 the CPU tier's
    flops bind and the two tiers stop being parallel lines.

    KV is held resident even for streamed layers: weights are reread every step
    so they have to cross the bus, the KV cache does not. That's a choice this
    engine makes and llama.cpp's -ngl does not.
    """
    total_weights = weight_bytes(model, bits)
    total_kv = kv_bytes_per_token(model, seq_len, batch)
    flops = flops_per_step(model, batch)

    if offload_tier.kv_follows_weights:
        off_bytes = (total_weights + total_kv) * offload_frac
        res_bytes = (total_weights + total_kv) * (1 - offload_frac)
    else:
        off_bytes = total_weights * offload_frac
        res_bytes = total_weights * (1 - offload_frac) + total_kv

    t_res = tier_time(res_bytes, flops * (1 - offload_frac), TIER_RESIDENT)
    t_off = tier_time(off_bytes, flops * offload_frac, offload_tier)
    return t_res + t_off

def predict_throughput(model: ModelConfig, bits: int, offload_frac: float, seq_len: int, batch: int,
                       offload_tier: Tier = TIER_STREAM) -> float:
    """Predicted decode throughput in tokens/sec = batch / t_token."""
    # batch / (s / token * batch) = token / s
    return batch / decode_time_per_token(model, bits, offload_frac, seq_len, batch, offload_tier)


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


def sweep_offload_fraction(model: ModelConfig, bits: int, fracs_list: list[float], seq_len: int, batch: int,
                           offload_tier: Tier = TIER_STREAM) -> dict:
    """
    Force offload_frac from 0 to 1 and predict tokens/sec at each point. The curve
    Step 2 overlays llama.cpp -ngl measurements onto. Returns 'offload_frac', 'tps'.
    """

    results = {"offload_frac": [], "tps": []}

    for frac in fracs_list:
        tps = predict_throughput(model, bits, frac, seq_len, batch, offload_tier)
        results["offload_frac"].append(frac)
        results["tps"].append(tps)

    return results


def sweep_batch(model: ModelConfig, bits: float, batches: list[int], seq_len: int, offload_frac: float,
                offload_tier: Tier = TIER_STREAM) -> dict:
    """
    Step 2 measured one point on this axis (batch 1) and the whole argument for
    the engine lives on the rest of it. Returns 'batch' and 'tps' for one tier;
    run it per tier and compare the curves.
    """

    results = {"batch": [], "tps": []}

    for batch in batches:
        results["batch"].append(batch)
        results["tps"].append(predict_throughput(model, bits, offload_frac, seq_len, batch, offload_tier))

    return results


def compute_bound_batch(model: ModelConfig, bits: float, tier: Tier, offload_frac: float = 1.0) -> float:
    """
    The batch where a tier stops being bandwidth bound and starts being compute
    bound, i.e. where its throughput stops climbing and flattens:

        bytes / BW  ==  2 * params * b / FLOPS

    KV is left out so this is a property of the tier and the weights alone.
    """
    t_bw = weight_bytes(model, bits) * offload_frac / (tier.bw_gb_s * 1e9)
    return t_bw * (tier.tflops * 1e12) / (2 * model.num_params * offload_frac)


def crossover_batch(model: ModelConfig, bits: float, seq_len: int, offload_frac: float = 1.0,
                    tier_a: Tier = TIER_CPU, tier_b: Tier = TIER_STREAM,
                    max_batch: int = 4096) -> float | None:
    """
    b*: the batch where tier_b overtakes tier_a. Below it llama.cpp's CPU offload
    is the right call and streaming to the GPU loses, above it the reverse. Found
    by scan rather than algebra because each curve has a knee in it.

    Returns None if they never cross below max_batch.
    """
    if predict_throughput(model, bits, offload_frac, seq_len, 1, tier_b) > \
       predict_throughput(model, bits, offload_frac, seq_len, 1, tier_a):
        return 1.0

    for batch in range(2, max_batch + 1):
        if predict_throughput(model, bits, offload_frac, seq_len, batch, tier_b) > \
           predict_throughput(model, bits, offload_frac, seq_len, batch, tier_a):
            return float(batch)

    return None

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
    plt.savefig(OUT_DIR / "two-tier-roofline.png", dpi=150)
    print(f"Saved {OUT_DIR / 'two-tier-roofline.png'}")
