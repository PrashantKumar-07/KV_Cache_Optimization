#!/bin/bash
# Improvement batch — TieredKV-only ablations vs clean baselines.
# Same revision/seed/ctx as clean reruns. Short tasks only; gov follows winners.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
SHORT="multifieldqa_en qasper narrativeqa hotpotqa triviaqa"
COMMON="--model $MODEL --methods tieredkv --budget 1024 --seed 0 --device cuda"
run() {
  echo "===== $(date) START $1 ====="
  nice -n 10 python experiments/longbench_eval.py $COMMON --tasks $2 --max-samples $3 $4 --json "$1"
  echo "===== $(date) DONE $1 (exit $?) ====="
}
# I1: fp32 diagnostic — isolates bf16-vs-fp32 effect on TieredKV accuracy
run results/improve_fp32_short.json      "$SHORT" 25 "--tiered-vram-budget 1024 --tiered-stt-budget 2048 --tiered-kv-dtype float32"
# I2: expose64 — vram960 + expose64 = 1024 attended (same attention as baselines)
run results/improve_expose64_short.json  "$SHORT" 25 "--tiered-vram-budget 960 --tiered-stt-budget 2048 --tiered-stt-expose 64 --tiered-kv-dtype bfloat16"
# I3: expose128 — vram896 + expose128 = 1024 attended
run results/improve_expose128_short.json "$SHORT" 25 "--tiered-vram-budget 896 --tiered-stt-budget 2048 --tiered-stt-expose 128 --tiered-kv-dtype bfloat16"
echo "===== $(date) ALL DONE ====="
