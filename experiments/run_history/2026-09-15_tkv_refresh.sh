#!/bin/bash
# TieredKV-only refresh under final code (STT 50/12.5 tentpole, drop_cost fix):
# F1 guaranteed identical (accounting-only change); refreshes modeled columns.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
SHORT="multifieldqa_en qasper narrativeqa hotpotqa triviaqa"
run() {
  echo "===== $(date) START $1 ====="
  nice -n 10 python experiments/longbench_eval.py --model $MODEL --methods tieredkv \
    --tasks $2 --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16 \
    $4 --max-samples $3 --json "$1"
  echo "===== $(date) DONE $1 (exit $?) ====="
}
run results/rf_vram_short.json    "$SHORT"   25 "--tiered-vram-budget 1024 --tiered-stt-budget 2048"
run results/rf_vram_gov.json      "gov_report" 10 "--tiered-vram-budget 1024 --tiered-stt-budget 2048"
run results/rf_att_short.json     "$SHORT"   25 "--tiered-vram-budget 992 --tiered-stt-budget 2048 --tiered-stt-expose 32"
run results/rf_att_gov.json       "gov_report" 10 "--tiered-vram-budget 992 --tiered-stt-budget 2048 --tiered-stt-expose 32"
run results/rf_tot_short.json     "$SHORT"   25 "--tiered-vram-budget 341 --tiered-stt-budget 683"
run results/rf_tot_gov.json       "gov_report" 10 "--tiered-vram-budget 341 --tiered-stt-budget 683"
echo "===== $(date) ALL DONE ====="
