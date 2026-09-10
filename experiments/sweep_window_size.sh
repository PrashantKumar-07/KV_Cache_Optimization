#!/bin/bash
# sweep_window_size.sh — ablation: sliding-window (recent-token) size, at
# equal-VRAM TieredKV config. Usage: MODEL=... ./sweep_window_size.sh [tasks...]
set -e
MODEL="${MODEL:?set MODEL=/path/to/model}"
# Default task list is the same 3-task subset as sweep_stt_budget.sh.
TASKS="${@:-multifieldqa_en hotpotqa qasper}"
SAMPLES="${SAMPLES:-25}"
SEED="${SEED:-0}"
# max_samples and seed are part of the config fingerprint, so a
# checkpoint from a different sample count will refuse to resume
# rather than blending two sample sets into one table.

for w in 32 64 128 256; do
  echo "=== recent_size(window)=$w ==="
  python experiments/longbench_eval.py --model "$MODEL" \
    --tasks $TASKS --methods tieredkv --max-samples "$SAMPLES" --budget 1024 \
    --tiered-vram-budget 1024 --tiered-stt-budget 2048 --recent-size "$w" --seed "$SEED" \
    --json "results/sweep_window/w${w}.json"
done
