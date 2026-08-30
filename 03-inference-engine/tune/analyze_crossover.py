"""
Turns a crossover.sh sweep into b*, the batch where GPU op offload starts paying.

For each prompt size the sweep gives two throughputs: offload forced on, and
offload forced off. Their ratio starts below 1 (the copy costs more than the GPU
saves) and ends above it. b* is where the ratio crosses 1, found by taking the
median over repetitions and interpolating linearly between the two prompt sizes
that bracket the crossing. Median rather than mean because llama-bench
occasionally throws a single very slow repetition -- see the stddev of 21.5 on a
mean of 44.5 at ngl=8 / NEVER / pp20 in the Q4_K_M data.

This is the script behind the three numbers the README quotes:

    Q4_K_M, -ngl 8/16/24 -> 15.3 / 15.5 / 15.6   (placement barely moves it)
    Q8_0,   -ngl 16      -> 28.7                 (the quant nearly doubles it)

against llama.cpp's default of 32 for all four.

    python3 -m tune.analyze_crossover
    python3 -m tune.analyze_crossover data/crossover_q4_k_m.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

DATA_DIR = Path(__file__).parent / "data"
LLAMACPP_MIN_BATCH = 32


def load(path: Path) -> dict:
    samples = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["ngl"]), int(row["n_prompt"]), row["mode"])
            samples[key].append(float(row["avg_ts"]))
    return {k: median(v) for k, v in samples.items()}


def crossover(tps: dict, ngl: int) -> tuple:
    """
    b* for one -ngl, plus the per-prompt table it was read off.

    Rows are (n_prompt, offload_tps, cpu_tps, ratio). The crossing is linear in
    the ratio between the bracketing prompt sizes; anything fancier would be
    reading structure into seven points that aren't there.
    """
    prompts = sorted({p for (n, p, _) in tps if n == ngl})
    rows = []
    for p in prompts:
        on = tps.get((ngl, p, "ALWAYS"))
        off = tps.get((ngl, p, "NEVER"))
        if on is None or off is None:
            continue
        rows.append((p, on, off, on / off))

    b_star = None
    for (p_lo, _, _, r_lo), (p_hi, _, _, r_hi) in zip(rows, rows[1:]):
        if r_lo < 1.0 <= r_hi:
            b_star = p_lo + (p_hi - p_lo) * (1.0 - r_lo) / (r_hi - r_lo)
            break
    return b_star, rows


def report(path: Path) -> None:
    tps = load(path)
    print(f"\n{path.name}")
    for ngl in sorted({n for (n, _, _) in tps}):
        b_star, rows = crossover(tps, ngl)
        print(f"\n  -ngl {ngl}")
        print(f"  {'prompt':>7} {'offload':>9} {'cpu-only':>9} {'ratio':>7}")
        for p, on, off, ratio in rows:
            print(f"  {p:>7} {on:>9.1f} {off:>9.1f} {ratio:>7.2f}")
        if b_star is None:
            print("  no crossing in the swept range")
            continue
        print(f"  b* = {b_star:.1f}   (llama.cpp default: {LLAMACPP_MIN_BATCH})")

        # What the default costs, read straight off the table: the largest gap
        # between the two curves at a prompt size the default still runs on CPU.
        missed = [(p, ratio) for p, _, _, ratio in rows if b_star <= p < LLAMACPP_MIN_BATCH]
        if missed:
            p, gain = max(missed, key=lambda t: t[1])
            print(f"  the default leaves {gain:.2f}x on the table at pp{p}")


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] or sorted(DATA_DIR.glob("crossover_*.csv"))
    for path in paths:
        report(path)
    print()
