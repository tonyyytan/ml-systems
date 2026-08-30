#!/usr/bin/env bash
# Measures where GPU op offload overtakes CPU compute, for one model.
#
# At each prompt size, run llama-bench twice: once with op offload forced ON
# (GGML_OP_OFFLOAD_MIN_BATCH=1) and once with it forced OFF (=99999). The batch
# where the two curves cross is b*, the value GGML_OP_OFFLOAD_MIN_BATCH should
# have. llama.cpp defaults it to 32 on every machine and every model.
#
# The two configs are INTERLEAVED inside each repetition. A sequential sweep of
# the threshold looks like it works and does not: the laptop heats up over the
# run, so the configs measured late lose 20% for reasons that have nothing to do
# with offload. Alternating cancels the drift instead of averaging over it. The
# confounded first attempt is kept in data/op_offload_naive_sweep.csv.
#
#   ./crossover.sh -m model.gguf -o data/crossover_q4_k_m.csv -g "8 16 24"
#
# Feed the output to analyze_crossover.py.
set -euo pipefail

LLAMA_BENCH="${LLAMA_BENCH:-$HOME/llama.cpp/build/bin/llama-bench}"
MODEL=""
OUT=""
NGLS="16"
PROMPTS="8 12 16 20 24 32 48"
REPS=3
BENCH_REPS=5
OFFLOAD_ON=1
OFFLOAD_OFF=99999

usage() { echo "usage: $0 -m model.gguf -o out.csv [-g \"8 16 24\"] [-p \"8 12 16\"] [-r 3]" >&2; exit 1; }

while getopts "m:o:g:p:r:" opt; do
    case $opt in
        m) MODEL=$OPTARG ;;
        o) OUT=$OPTARG ;;
        g) NGLS=$OPTARG ;;
        p) PROMPTS=$OPTARG ;;
        r) REPS=$OPTARG ;;
        *) usage ;;
    esac
done
[ -n "$MODEL" ] && [ -n "$OUT" ] || usage
[ -x "$LLAMA_BENCH" ] || { echo "llama-bench not at $LLAMA_BENCH (set LLAMA_BENCH)" >&2; exit 1; }

read -ra PROMPT_LIST <<< "$PROMPTS"
PROMPT_CSV=$(IFS=,; echo "${PROMPT_LIST[*]}")

echo "ngl,rep,mode,n_prompt,avg_ts,stddev_ts" > "$OUT"

for ngl in $NGLS; do
    for rep in $(seq 1 "$REPS"); do
        for mode in ALWAYS NEVER; do
            if [ "$mode" = ALWAYS ]; then
                threshold=$OFFLOAD_ON
            else
                threshold=$OFFLOAD_OFF
            fi

            # llama-bench csv: avg_ts is field 39, stddev_ts is 40. One row per
            # prompt size, in the order they were requested.
            GGML_OP_OFFLOAD_MIN_BATCH=$threshold "$LLAMA_BENCH" \
                -m "$MODEL" -ngl "$ngl" -p "$PROMPT_CSV" -n 0 -r "$BENCH_REPS" -o csv 2>/dev/null \
            | awk -F, -v ngl="$ngl" -v rep="$rep" -v mode="$mode" -v prompts="$PROMPTS" '
                BEGIN { split(prompts, p, " ") }
                NR > 1 { gsub(/"/, ""); print ngl "," rep "," mode "," p[++i] "," $39 "," $40 }' >> "$OUT"
        done
    done
    echo "ngl=$ngl done" >&2
done

echo "wrote $OUT" >&2
