#!/bin/bash
# sweep_stt_budget.sh — ablation: STT-RAM (slow-tier) size at fixed VRAM=1024.
# Answers "how much does the slow tier's size matter?" for the equal-VRAM
# headline comparison. Usage: MODEL=... ./sweep_stt_budget.sh [tasks...]
set -e
MODEL="${MODEL:?set MODEL=/path/to/model}"
# Default task list is the 3 tasks the README ablation table reports.
TASKS="${@:-multifieldqa_en hotpotqa qasper}"
SAMPLES="${SAMPLES:-25}"
SEED="${SEED:-0}"
# max_samples and seed are part of the config fingerprint, so a
# checkpoint from a different sample count will refuse to resume
# rather than blending two sample sets into one table.

for stt in 512 1024 2048 3072; do
  echo "=== stt_budget=$stt (vram fixed at 1024) ==="
  python experiments/longbench_eval.py --model "$MODEL" \
    --tasks $TASKS --methods tieredkv --max-samples "$SAMPLES" --budget 1024 \
    --tiered-vram-budget 1024 --tiered-stt-budget "$stt" --seed "$SEED" \
    --json "results/sweep_stt/stt${stt}.json"
done
