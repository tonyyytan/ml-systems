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
# TODO: fill from Step 0 on the machine this actually runs on.
VRAM_BW_GB_S = None   # resident slope,  ~272 on RTX 5060 / ~128 on T1000
PCIE_BW_GB_S = None   # offload slope,   MEASURED, pinned H2D. expect ~16-32
VRAM_CAPACITY_GB = None   # 8 on 5060, 4 on T1000. sets where the cliff sits.


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
LLAMA3_8B = ModelConfig(name="llama-3-8b", num_layers=32, hidden_dim=4096,
                        num_params=8_030_000_000, num_attention_heads = 32, num_kv_heads = 8)


def bytes_per_param(bits: int) -> float:
    """Bytes each weight crosses the bus as, at a given precision (16/8/4)."""
    return bits / 8


def weight_bytes(model: ModelConfig, bits: int) -> float:
    """Total weight bytes streamed to produce one token (decode reads all of them)."""
    return model.num_params * bytes_per_param(bits)


def kv_bytes_per_token(model: ModelConfig, bits: int, seq_len: int,
                       batch: int) -> float:
    """KV-cache bytes read per decode step. Grows with seq_len and batch."""
    head_dim = model.hidden_dim / model.num_attention_heads
    return 2 * head_dim * model.num_kv_heads * model.num_layers * seq_len * batch * bytes_per_param(bits)


def decode_time_per_token(model: ModelConfig, bits: int, offload_frac: float,
                          seq_len: int, batch: int) -> float:
    """
    The two-tier equation. offload_frac in [0,1] is the fraction of layers that
    spill to RAM; those bytes pay the PCIE slope, the rest pay the VRAM slope.

        t = resident_bytes / VRAM_BW  +  offloaded_bytes / PCIE_BW
    """
    total_weights = weight_bytes(model, bits)
    total_kv = kv_bytes_per_token(model, bits, seq_len, batch)

    ram_bytes = (total_weights * offload_frac) + (total_kv * offload_frac)
    vram_bytes = (total_weights * (1 - offload_frac)) + (total_kv * (1 - offload_frac))

    time = vram_bytes / VRAM_BW + ram_bytes / PCIE_BW
    return time

def predict_throughput(model: ModelConfig, bits: int, offload_frac: float,
                       seq_len: int, batch: int) -> float:
    """Predicted decode throughput in tokens/sec = batch / t_token."""

    return batch / decode_time_per_token(model, bits, offload_frac, seq_len, batch)


def offload_fraction(model: ModelConfig, bits: int, vram_gb: float) -> float:
    """
    How much of the model spills given a VRAM budget. This is where the cliff
    lives: at the precision that just fits, offload_frac drops to 0 and the
    PCIE term vanishes from decode_time_per_token.
    """

    return min(1.0, max(0.0, (weight_bytes(model, bits) / 1e9 - vram_gb) / (weight_bytes(model, bits) / 1e9)))

def arithmetic_intensity(model: ModelConfig, bits: int, seq_len: int, batch: int) -> float:
    """
    FLOP per byte at a given batch. At batch 1 this is ~1-2 (memory bound, the
    wall). The point of the sweep is finding the batch where compute finally
    hides a PCIE transfer, i.e. where prefetch stops being free money lost.
    """

    return 2 * model.num_params * batch / (weight_bytes(model, bits) + kv_bytes_per_token(model, bits, seq_len, batch))

# --- sweeps -----------------------------------------------------------------

def sweep_precision(model: ModelConfig, bits_list: list[int], vram_gb: float,
                    seq_len: int, batch: int) -> dict:
    """
    The cliff sweep. For each precision: does it fit, what's offload_frac, and
    the predicted tokens/sec. Returns lists keyed 'bits', 'offload_frac',
    'tokens_per_sec', 'model_gb'.
    """
    # TODO: loop bits -> offload_fraction -> predict_throughput.
    raise NotImplementedError


def sweep_offload_fraction(model: ModelConfig, bits: int, fracs: list[float],
                           seq_len: int, batch: int) -> dict:
    """
    The continuous version: force offload_frac from 0 to 1 and predict
    tokens/sec at each point. THIS is the curve Step 2 overlays llama.cpp
    -ngl measurements onto. Returns 'offload_frac', 'tokens_per_sec'.
    """
    
    # TODO: loop fracs -> predict_throughput. Keep the arg the same knob
    #       -ngl exposes so the overlay is apples to apples.
    raise NotImplementedError


# --- plot -------------------------------------------------------------------

def plot_two_tier(precision_sweep: dict, offload_sweep: dict) -> None:
    """
    The artifact. Two panels:
      (a) tokens/sec vs precision, with the VRAM boundary drawn as a vertical
          line so the step at the fit/no-fit crossing is visible as a cliff.
      (b) tokens/sec vs offload fraction, the line Step 2 validates against.
    Mark VRAM_CAPACITY_GB and annotate where each precision lands.
    """
    # TODO: implement. Match the 01 plot style (log axes where it helps,
    #       series colors, annotated points). Save two-tier-roofline.png.
    raise NotImplementedError


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
