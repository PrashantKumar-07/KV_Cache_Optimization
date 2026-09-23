#!/bin/bash
# Overnight clean reruns — TieredKV publication baseline.
# Branch: fix/publication-ready. One code revision, fixed seed, max_ctx=31500,
# bf16 TieredKV residency (footprint-honest), 25 samples/short-task, 10 gov_report.
# GPU 1 only (GPU 0 untouched), checkpointed resume per job.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
SHORT="multifieldqa_en qasper narrativeqa hotpotqa triviaqa"
METHODS="full streamingllm h2o snapkv tieredkv"
COMMON="--model $MODEL --methods $METHODS --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16"

run() {  # $1=json $2=tasks $3=samples $4=extra tiered flags
  echo "===== $(date) START $1 ====="
  nice -n 10 python experiments/longbench_eval.py $COMMON --tasks $2 --max-samples $3 $4 --json "$1"
  echo "===== $(date) DONE $1 (exit $?) ====="
}

# A. equal-VRAM headline-style: vram=1024 + stt=2048, unbounded exposure
run results/clean_equal_vram_short.json    "$SHORT"   25 "--tiered-vram-budget 1024 --tiered-stt-budget 2048"
run results/clean_equal_vram_gov.json      "gov_report" 10 "--tiered-vram-budget 1024 --tiered-stt-budget 2048"
# B. equal-attended: vram=992 + expose=32 -> 1024 attended, same as baselines
run results/clean_equal_attended_short.json "$SHORT"   25 "--tiered-vram-budget 992 --tiered-stt-budget 2048 --tiered-stt-expose 32"
run results/clean_equal_attended_gov.json   "gov_report" 10 "--tiered-vram-budget 992 --tiered-stt-budget 2048 --tiered-stt-expose 32"
# C. equal-total: vram=341 + stt=683 (sums to budget 1024)
run results/clean_equal_total_short.json   "$SHORT"   25 "--tiered-vram-budget 341 --tiered-stt-budget 683"
run results/clean_equal_total_gov.json     "gov_report" 10 "--tiered-vram-budget 341 --tiered-stt-budget 683"
echo "===== $(date) ALL DONE ====="
