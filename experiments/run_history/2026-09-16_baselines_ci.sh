#!/bin/bash
# Baselines refresh under final code (sample_scores for CIs).
# Baselines are tier-config-independent: one run backfills all three finals.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
echo "===== $(date) START base_short ====="
nice -n 10 python experiments/longbench_eval.py --model $MODEL \
  --methods full streamingllm h2o snapkv --tasks multifieldqa_en qasper narrativeqa hotpotqa triviaqa \
  --budget 1024 --seed 0 --device cuda --max-samples 25 --json results/baseci_short.json
echo "===== $(date) DONE base_short (exit $?) ====="
echo "===== $(date) START base_gov ====="
nice -n 10 python experiments/longbench_eval.py --model $MODEL \
  --methods full streamingllm h2o snapkv --tasks gov_report \
  --budget 1024 --seed 0 --device cuda --max-samples 10 --json results/baseci_gov.json
echo "===== $(date) DONE base_gov (exit $?) ====="
echo "===== $(date) ALL DONE ====="
