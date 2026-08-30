"""
The deliverable: measure the machine, run the roofline, print the flags.

llama.cpp already has every mechanism this project needs. -ngl places layers,
-ot places individual tensors by regex, --n-cpu-moe places experts, and
GGML_OP_OFFLOAD_MIN_BATCH decides when a big-enough batch gets shipped to the
gpu instead of computed on the cpu. What it does not have is a policy. Nothing
tells you what to set them to, so in practice everybody runs the defaults, and
the offload threshold defaults to the literal integer 32 on every machine, every
model and every tensor shape.

This is the policy. It takes machine.json from measure_machine.py, runs the
tier roofline over it, and prints the flags for one model at one context length:

    $ python3 -m tune.autotune -m ~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    GGML_OP_OFFLOAD_MIN_BATCH=15 llama-cli -m ... -ngl 16 -c 2048

The threshold is the interesting one. It is not a better constant -- it moves
with the machine's pcie and cpu bandwidth AND with the model's bytes per weight,
which is why the measured value is 15.5 for Q4_K_M and 28.7 for Q8_0 on this
same laptop at the same -ngl. See tune/analyze_crossover.py.

--verify runs the answer against llama.cpp's defaults, alternating the two
configurations so thermal drift cannot favour either, and prints the delta.
That is the number that says whether any of this was worth doing.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median

from roofline import roofline2 as r2

TUNE_DIR = Path(__file__).parent
DEFAULT_MACHINE = TUNE_DIR / "machine.json"

# llama.cpp needs room for the compute buffers, the cuda context and the graph
# on top of the weights it is told to place. Chosen to be safe rather than tight:
# overshooting -ngl by one layer costs far more than leaving one on the cpu.
VRAM_RESERVE_GB = 1.2

LLAMACPP_MIN_BATCH = 32
VERIFY_REPS = 3
VERIFY_BENCH_REPS = 5


def load_machine(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"no {path}; run `python3 -m tune.measure_machine -m model.gguf` first")
    machine = json.loads(path.read_text())
    if "cpu_tflops" not in machine:
        sys.exit(f"{path} has no model probe; re-run measure_machine.py with -m")
    return machine


def apply_machine(machine: dict) -> None:
    """
    Point the roofline at this machine's constants instead of the ones baked
    into roofline2.py. Rebinding module globals rather than threading a config
    object through every function: roofline2 is a model of one machine at a
    time, and the tiers are derived from the constants, so they get rebuilt too.

    Only what the model reads at call time is set. PEAK_BW_GB_S and STREAM_EFF
    are consumed at import to derive VRAM_BW_GB_S, which is set here directly.
    """
    r2.VRAM_BW_GB_S = machine["vram_bw_gb_s"]
    r2.PCIE_BW_GB_S = machine["h2d_pinned_gb_s"]
    r2.CPU_BW_GB_S = machine["cpu_bw_gb_s"]
    r2.CPU_TFLOPS = machine["cpu_tflops"]
    r2.GPU_TFLOPS = machine["gpu_tflops"]
    r2.VRAM_CAPACITY_GB = machine["vram_capacity_gb"]

    r2.TIER_RESIDENT = r2.Tier("resident", r2.VRAM_BW_GB_S, r2.GPU_TFLOPS)
    r2.TIER_CPU = r2.Tier("cpu-offload", r2.CPU_BW_GB_S, r2.CPU_TFLOPS, kv_follows_weights=True)
    r2.TIER_STREAM = r2.Tier("pcie-stream", r2.PCIE_BW_GB_S, r2.GPU_TFLOPS)
    r2.TIER_STREAM_NAIVE = r2.Tier("pcie-stream (no prefetch)", r2.PCIE_BW_GB_S, r2.GPU_TFLOPS, overlap=False)


def model_stats(binary: str, model: str) -> dict:
    """
    Parameter count and file size for a gguf llama-bench has not already
    reported on. One token of prefill on the cpu, purely to read the header
    back out of the csv -- a couple of seconds.
    """
    cmd = [binary, "-m", model, "-ngl", "0", "-p", "1", "-n", "0", "-r", "1", "-o", "csv"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"could not read {model}:\n{proc.stderr.strip()}")
    lines = [l for l in proc.stdout.strip().splitlines() if l]
    row = dict(zip([h.strip('"') for h in lines[0].split(",")],
                   [v.strip('"') for v in lines[-1].split(",")]))
    n_params, size = int(row["model_n_params"]), int(row["model_size"])
    return {"filename": Path(row["model_filename"]).name, "type": row["model_type"],
            "n_params": n_params, "size_bytes": size,
            "bits_per_param": round(size * 8 / n_params, 2)}


def model_config(machine: dict, args) -> tuple[r2.ModelConfig, float]:
    """
    The model, and the bits each weight actually crosses the bus as. n_params
    and the file size are measured (llama-bench reports both), so bits/param is
    measured too -- 4.89 for Q4_K_M, not the 4 the name suggests. The shape
    fields default to llama-3-8b and are printed so a wrong assumption is
    visible rather than silent.

    The bandwidth and compute constants are properties of the machine and carry
    over, but bits/param is a property of THIS gguf -- which is the whole reason
    the threshold cannot be cached per device -- so a model that was not the one
    measured gets read directly.
    """
    m = machine["model"]
    if m["filename"] != Path(args.model).name:
        m = model_stats(args.llama_bench, args.model)
    config = r2.ModelConfig(name=m["type"], num_layers=args.n_layers, hidden_dim=args.hidden_dim,
                            num_params=m["n_params"], num_attention_heads=args.n_heads,
                            num_kv_heads=args.n_kv_heads)
    return config, m["bits_per_param"]


def choose_ngl(model: r2.ModelConfig, bits: float, seq_len: int, batch: int, vram_gb: float) -> int:
    """
    How many layers fit. A resident layer costs its weights plus its share of
    the kv cache, since under -ngl the cache follows the layer.

    Weights are divided evenly across layers, which is not exactly right --
    the token embedding and the output head are not layers and llama.cpp counts
    them separately -- so this is a layer or so conservative on small models.
    """
    budget_bytes = max(0.0, vram_gb - VRAM_RESERVE_GB) * 1e9
    per_layer = (r2.weight_bytes(model, bits) +
                 r2.kv_bytes_per_token(model, seq_len, batch)) / model.num_layers
    return max(0, min(model.num_layers, int(budget_bytes // per_layer)))


def recommend(model: r2.ModelConfig, bits: float, seq_len: int, batch: int, vram_gb: float,
              force_ngl: int | None = None) -> dict:
    ngl = choose_ngl(model, bits, seq_len, batch, vram_gb) if force_ngl is None else force_ngl
    offload_frac = 1 - ngl / model.num_layers

    if offload_frac == 0:
        return {"ngl": ngl, "offload_frac": 0.0, "min_batch": None,
                "note": "the model fits; the offload threshold never fires"}

    b_star = r2.crossover_batch(model, bits, seq_len, offload_frac, r2.TIER_CPU, r2.TIER_STREAM)
    if b_star is None:
        return {"ngl": ngl, "offload_frac": offload_frac, "min_batch": None,
                "note": "cpu offload wins at every batch on this machine; disable op offload"}

    def at(tier, batch):
        return r2.predict_throughput(model, bits, offload_frac, seq_len, int(batch), tier)

    rec = {"ngl": ngl, "offload_frac": offload_frac, "min_batch": int(b_star),
           "tps_cpu": at(r2.TIER_CPU, b_star), "tps_stream": at(r2.TIER_STREAM, b_star)}

    # only meaningful when the default sits above b*, which is the case worth
    # reporting: there is a band the default spends on the wrong tier.
    if b_star < LLAMACPP_MIN_BATCH:
        rec["default_gain"] = (at(r2.TIER_STREAM, LLAMACPP_MIN_BATCH - 1) /
                               at(r2.TIER_CPU, LLAMACPP_MIN_BATCH - 1))
    return rec


def report(machine: dict, model: r2.ModelConfig, bits: float, args, rec: dict) -> None:
    print(f"\nmachine   {machine['device']}, {machine['vram_capacity_gb']:.1f} GB")
    link = machine.get("pcie_link")
    if link:
        print(f"          pcie gen{link['gen_current']} x{link['width_current']} "
              f"(max gen{link['gen_max']} x{link['width_max']})")
    print(f"          {r2.VRAM_BW_GB_S:.0f} GB/s vram, {r2.PCIE_BW_GB_S:.1f} GB/s pcie, "
          f"{r2.CPU_BW_GB_S:.0f} GB/s cpu")
    print(f"          {r2.CPU_TFLOPS} / {r2.GPU_TFLOPS} TFLOP/s cpu / gpu")

    print(f"\nmodel     {model.name}, {model.num_params / 1e9:.1f}B params at {bits} bits/param")
    print(f"          {model.num_layers} layers, hidden {model.hidden_dim}, "
          f"{model.num_attention_heads}/{model.num_kv_heads} heads  (assumed, override with --n-layers etc)")
    print(f"          context {args.ctx}, batch {args.batch}")

    print(f"\nplacement {rec['ngl']}/{model.num_layers} layers resident "
          f"({rec['offload_frac']:.0%} offloaded)")

    if rec["min_batch"] is None:
        print(f"threshold {rec['note']}")
    else:
        print(f"threshold b* = {rec['min_batch']} against llama.cpp's default of "
              f"{LLAMACPP_MIN_BATCH}")
        if "default_gain" in rec:
            print(f"          at b={LLAMACPP_MIN_BATCH - 1} the default costs "
                  f"{rec['default_gain']:.2f}x")

    print("\nflags")
    env = "" if rec["min_batch"] is None else f"GGML_OP_OFFLOAD_MIN_BATCH={rec['min_batch']} "
    print(f"  {env}llama-cli -m {Path(args.model).name} -ngl {rec['ngl']} -c {args.ctx}")

    # Claim (2) in the readme -- that the right threshold differs between the
    # mlp and attention tensors -- is not measured yet, so no -ot regex is
    # emitted. Guessing one would be the exact failure this tool exists to fix.
    print("\n  no -ot emitted: per-tensor thresholds are predicted to differ but not yet measured")


def llama_bench(binary: str, model: str, ngl: int, prompts: list[int], min_batch: int | None) -> dict:
    cmd = [binary, "-m", model, "-ngl", str(ngl), "-p", ",".join(map(str, prompts)),
           "-n", "0", "-r", str(VERIFY_BENCH_REPS), "-o", "csv"]
    env = dict(os.environ)
    if min_batch is not None:
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = str(min_batch)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.exit(f"llama-bench failed:\n{proc.stderr.strip()}")

    lines = [l for l in proc.stdout.strip().splitlines() if l]
    avg_ts = [h.strip('"') for h in lines[0].split(",")].index("avg_ts")
    return {p: float(row.split(",")[avg_ts].strip('"')) for p, row in zip(prompts, lines[1:])}


def verify(args, rec: dict) -> None:
    """
    The tuned flags against llama.cpp's defaults, on the real binary. Configs
    alternate inside each repetition for the same reason crossover.sh does it:
    a laptop measured over ten minutes is a different machine at the end of the
    run than at the start, and a sequential comparison quietly charges that
    drift to whichever config went last.
    """
    if rec["min_batch"] is None:
        sys.exit("nothing to verify: no threshold was recommended")

    prompts = sorted({rec["min_batch"], (rec["min_batch"] + LLAMACPP_MIN_BATCH) // 2,
                      LLAMACPP_MIN_BATCH - 1, LLAMACPP_MIN_BATCH * 2})
    print(f"\nverifying at prompts {prompts}, {VERIFY_REPS} alternating reps", file=sys.stderr)

    samples = {"tuned": {}, "default": {}}
    for rep in range(VERIFY_REPS):
        for label, min_batch in [("tuned", rec["min_batch"]), ("default", LLAMACPP_MIN_BATCH)]:
            print(f"  rep {rep + 1} {label}", file=sys.stderr)
            for p, tps in llama_bench(args.llama_bench, args.model, rec["ngl"], prompts, min_batch).items():
                samples[label].setdefault(p, []).append(tps)

    print(f"\n{'prompt':>7} {'tuned':>9} {'default':>9} {'gain':>7}")
    for p in prompts:
        tuned, default = median(samples["tuned"][p]), median(samples["default"][p])
        marker = "" if p < LLAMACPP_MIN_BATCH else "   (control: both offload here)"
        print(f"{p:>7} {tuned:>9.1f} {default:>9.1f} {tuned / default:>6.2f}x{marker}")


def main() -> None:
    ap = argparse.ArgumentParser(description="emit llama.cpp placement flags from a measured roofline")
    ap.add_argument("-m", "--model", required=True, help="the gguf these flags are for")
    ap.add_argument("--machine", type=Path, default=DEFAULT_MACHINE)
    ap.add_argument("-c", "--ctx", type=int, default=2048, help="context length to size the kv cache at")
    ap.add_argument("-b", "--batch", type=int, default=1, help="batch size to place for")
    ap.add_argument("--ngl", type=int, help="force a placement instead of choosing one, to ask "
                                            "what threshold that placement wants")
    ap.add_argument("--n-layers", type=int, default=32)
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-kv-heads", type=int, default=8)
    ap.add_argument("--verify", action="store_true", help="run the recommendation against the defaults")
    ap.add_argument("--llama-bench", default=str(Path.home() / "llama.cpp/build/bin/llama-bench"))
    args = ap.parse_args()

    machine = load_machine(args.machine)
    apply_machine(machine)
    model, bits = model_config(machine, args)

    rec = recommend(model, bits, args.ctx, args.batch, machine["vram_capacity_gb"], args.ngl)
    report(machine, model, bits, args, rec)

    if args.verify:
        verify(args, rec)
    print()


if __name__ == "__main__":
    main()
