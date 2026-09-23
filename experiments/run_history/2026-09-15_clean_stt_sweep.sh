#!/bin/bash
# Clean STT-budget sweep: vram=1024 fixed, STT 512/1024/2048/3072, bf16, seed 0,
# 5 short tasks x25, TieredKV-only, final code. Refreshes stale fig5/table4.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
mkdir -p results/clean_sweep_stt
for stt in 512 1024 2048 3072; do
  echo "===== $(date) START stt$stt ====="
  nice -n 10 python experiments/longbench_eval.py --model "$MODEL" \
    --tasks multifieldqa_en qasper narrativeqa hotpotqa triviaqa --methods tieredkv \
    --max-samples 25 --budget 1024 --tiered-vram-budget 1024 --tiered-stt-budget "$stt" \
    --tiered-kv-dtype bfloat16 --seed 0 --device cuda \
    --json "results/clean_sweep_stt/stt${stt}.json"
  echo "===== $(date) DONE stt$stt (exit $?) ====="
done
echo "===== $(date) ALL DONE ====="
