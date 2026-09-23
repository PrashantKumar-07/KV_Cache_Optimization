#!/bin/bash
# E1 ORACLE: infinite STT (never reclaim) — upper bound on shadow-cache savings.
# TieredKV-only, same revision/seed/ctx/dtype as clean runs. Short tasks x25.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
echo "===== $(date) START e1_oracle ====="
nice -n 10 python experiments/longbench_eval.py --model $MODEL --methods tieredkv \
  --tasks multifieldqa_en qasper narrativeqa hotpotqa triviaqa \
  --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16 \
  --tiered-vram-budget 1024 --tiered-stt-budget 100000 --max-samples 25 \
  --json results/e1_oracle_short.json
echo "===== $(date) DONE e1_oracle (exit $?) ====="
