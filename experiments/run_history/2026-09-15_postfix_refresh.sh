#!/bin/bash
# Post-starvation-fix refresh: TieredKV-only, equal-VRAM 1024+2048, bf16, seed 0.
# Baselines already clean; only TieredKV numbers refresh.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
COMMON="--model $MODEL --methods tieredkv --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16 --tiered-vram-budget 1024 --tiered-stt-budget 2048"
echo "===== $(date) START postfix_hotpotqa ====="
nice -n 10 python experiments/longbench_eval.py $COMMON --tasks hotpotqa --max-samples 25 --json results/postfix_hotpotqa.json
echo "===== $(date) DONE postfix_hotpotqa (exit $?) ====="
echo "===== $(date) START postfix_short ====="
nice -n 10 python experiments/longbench_eval.py $COMMON --tasks multifieldqa_en qasper narrativeqa triviaqa --max-samples 25 --json results/postfix_short.json
echo "===== $(date) DONE postfix_short (exit $?) ====="
echo "===== $(date) START postfix_gov ====="
nice -n 10 python experiments/longbench_eval.py $COMMON --tasks gov_report --max-samples 10 --json results/postfix_gov.json
echo "===== $(date) DONE postfix_gov (exit $?) ====="
echo "===== $(date) ALL DONE ====="
