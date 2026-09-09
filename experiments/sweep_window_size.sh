#!/bin/bash
# sweep_window_size.sh — ablation: sliding-window (recent-token) size, at
# equal-VRAM TieredKV config. Usage: MODEL=... ./sweep_window_size.sh [tasks...]
set -e
MODEL="${MODEL:?set MODEL=/path/to/model}"
TASKS="${@:-multifieldqa_en hotpotqa triviaqa qasper narrativeqa}"
SAMPLES="${SAMPLES:-15}"

for w in 32 64 128 256; do
  echo "=== recent_size(window)=$w ==="
  python experiments/longbench_eval.py --model "$MODEL" \
    --tasks $TASKS --methods tieredkv --max-samples "$SAMPLES" --budget 1024 \
    --tiered-vram-budget 1024 --tiered-stt-budget 2048 --recent-size "$w" \
    --json "results/sweep_window/w${w}.json"
done
