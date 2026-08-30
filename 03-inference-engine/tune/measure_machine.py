"""
Step 0: measure every constant the roofline is built on, and write them down.

roofline2.py carries seven numbers that decide every prediction it makes. Until
this script existed they were comments, each one traceable to a run I did once
by hand and typed in. That is fine for one laptop and useless for a tool that is
supposed to work on someone else's -- so this measures all seven and emits
machine.json, which autotune.py reads.

    constant          where it comes from
    ---------------   ----------------------------------------------------
    PEAK_BW_GB_S      bus width x memory clock, off the device itself
    VRAM_CAPACITY_GB  the device again; sets where the cliff falls
    PCIE_BW_GB_S      pinned host-to-device sweep (tune/h2d.cu)
    STREAM_EFF        llama.cpp decode, fully resident, vs that peak
    CPU_BW_GB_S       llama.cpp decode at -ngl 0, weights read out of DDR5
    CPU_TFLOPS        llama.cpp prefill at -ngl 0, op offload off
    GPU_TFLOPS        llama.cpp prefill, fully resident

The compute ceilings come from llama.cpp's prefill rather than a synthetic gemm
sweep on purpose: it is the same code path whose throughput the model is trying
to predict, so it prices the real thing including its inefficiencies. Effective
flops are 2 * params * tokens_per_sec, and the bandwidth numbers are
model_bytes * tokens_per_sec, since decode reads every weight once per token.

    python3 -m tune.measure_machine                     # gpu only, ~30s
    python3 -m tune.measure_machine -m model.gguf       # everything, ~5 min
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TUNE_DIR = Path(__file__).parent
H2D_SRC = TUNE_DIR / "h2d.cu"
H2D_BIN = TUNE_DIR / "h2d"
DEFAULT_OUT = TUNE_DIR / "machine.json"

ARCH = "sm_120"
PREFILL_TOKENS = 512
DECODE_TOKENS = 128
BENCH_REPS = 3
OFFLOAD_OFF = "99999"


def run(cmd: list[str], **kwargs) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}\n{proc.stderr.strip()}")
    return proc.stdout


def measure_gpu() -> dict:
    """Build and run the CUDA probe: device properties plus the PCIe sweep."""
    if not shutil.which("nvcc"):
        sys.exit("nvcc not found; the pcie slope is not optional")
    print("building tune/h2d.cu", file=sys.stderr)
    run(["nvcc", "-O3", "-std=c++17", f"-arch={ARCH}", "-o", str(H2D_BIN), str(H2D_SRC)])
    print("measuring host-to-device bandwidth", file=sys.stderr)
    return json.loads(run([str(H2D_BIN)]))


def measure_pcie_link() -> dict:
    """
    The negotiated link, which is not the link on the box. This laptop reports
    width.current=8 against width.max=16, and drops the current width further
    when the gpu is idle -- so a pcie number without the link state beside it
    cannot be compared against anyone else's.
    """
    if not shutil.which("nvidia-smi"):
        return {}
    fields = "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
    out = run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    gen_cur, gen_max, width_cur, width_max = (v.strip() for v in out.strip().split(","))
    return {"pcie_link": {"gen_current": int(gen_cur), "gen_max": int(gen_max),
                          "width_current": int(width_cur), "width_max": int(width_max)}}


def llama_bench(binary: str, model: str, ngl: int, n_prompt: int, n_gen: int,
                op_offload: bool = True) -> dict:
    """One llama-bench row, as a dict of its csv header to its value."""
    env_note = "" if op_offload else " (op offload off)"
    print(f"  llama-bench -ngl {ngl} -p {n_prompt} -n {n_gen}{env_note}", file=sys.stderr)
    cmd = [binary, "-m", model, "-ngl", str(ngl), "-p", str(n_prompt),
           "-n", str(n_gen), "-r", str(BENCH_REPS), "-o", "csv"]
    env = dict(os.environ)
    if not op_offload:
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = OFFLOAD_OFF
    out = run(cmd, env=env)
    lines = [l for l in out.strip().splitlines() if l]
    header = [h.strip('"') for h in lines[0].split(",")]
    row = [v.strip('"') for v in lines[-1].split(",")]
    return dict(zip(header, row))


def measure_model(binary: str, model: str, n_layers: int, peak_bw: float) -> dict:
    """
    The four llama.cpp probes. Everything here is one model on one machine, so
    the compute ceilings carry that model's inefficiencies with them -- which is
    the point, since it is that model the autotuner is being asked about.
    """
    print("probing llama.cpp (4 runs)", file=sys.stderr)

    cpu_pp = llama_bench(binary, model, 0, PREFILL_TOKENS, 0, op_offload=False)
    cpu_tg = llama_bench(binary, model, 0, 0, DECODE_TOKENS, op_offload=False)
    gpu_pp = llama_bench(binary, model, n_layers, PREFILL_TOKENS, 0)
    gpu_tg = llama_bench(binary, model, n_layers, 0, DECODE_TOKENS)

    n_params = int(cpu_pp["model_n_params"])
    model_bytes = int(cpu_pp["model_size"])

    def tflops(row): return 2 * n_params * float(row["avg_ts"]) / 1e12
    def bw(row): return model_bytes * float(row["avg_ts"]) / 1e9

    vram_bw = bw(gpu_tg)
    return {
        "model": {
            "filename": Path(cpu_pp["model_filename"]).name,
            "type": cpu_pp["model_type"],
            "n_params": n_params,
            "size_bytes": model_bytes,
            "bits_per_param": round(model_bytes * 8 / n_params, 2),
            "n_layers": n_layers,
        },
        "cpu_tflops": round(tflops(cpu_pp), 3),
        "gpu_tflops": round(tflops(gpu_pp), 1),
        "cpu_bw_gb_s": round(bw(cpu_tg), 1),
        "vram_bw_gb_s": round(vram_bw, 1),
        "stream_eff": round(vram_bw / peak_bw, 3),
        "raw_tokens_per_sec": {
            "cpu_prefill": round(float(cpu_pp["avg_ts"]), 2),
            "cpu_decode": round(float(cpu_tg["avg_ts"]), 2),
            "gpu_prefill": round(float(gpu_pp["avg_ts"]), 2),
            "gpu_decode": round(float(gpu_tg["avg_ts"]), 2),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="measure the constants the tier roofline runs on")
    ap.add_argument("-m", "--model", help="gguf to probe; without it only the gpu constants are measured")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--llama-bench", default=str(Path.home() / "llama.cpp/build/bin/llama-bench"))
    ap.add_argument("--n-layers", type=int, default=32, help="layers in the model, for the resident run")
    args = ap.parse_args()

    machine = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    machine.update(measure_gpu())
    machine.update(measure_pcie_link())

    if args.model:
        if not Path(args.llama_bench).is_file():
            sys.exit(f"llama-bench not at {args.llama_bench}")
        machine.update(measure_model(args.llama_bench, args.model, args.n_layers,
                                     machine["vram_peak_bw_gb_s_from_clocks"]))

    args.out.write_text(json.dumps(machine, indent=2) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)

    print(f"\n{'constant':<20} {'value':>10}")
    for key, label in [("vram_peak_bw_gb_s_from_clocks", "PEAK_BW_GB_S"),
                       ("vram_capacity_gb", "VRAM_CAPACITY_GB"),
                       ("h2d_pinned_gb_s", "PCIE_BW_GB_S"),
                       ("stream_eff", "STREAM_EFF"),
                       ("cpu_bw_gb_s", "CPU_BW_GB_S"),
                       ("cpu_tflops", "CPU_TFLOPS"),
                       ("gpu_tflops", "GPU_TFLOPS")]:
        if key in machine:
            print(f"{label:<20} {machine[key]:>10}")


if __name__ == "__main__":
    main()
