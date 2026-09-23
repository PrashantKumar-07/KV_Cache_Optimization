#!/bin/bash
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
SHORT="multifieldqa_en qasper narrativeqa hotpotqa triviaqa"
for L in 0.01 0.05; do
  echo "===== $(date) START wa075_$L ====="
  nice -n 10 python experiments/longbench_eval.py --model $MODEL --methods tieredkv \
    --tasks $SHORT --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16 \
    --tiered-vram-budget 1024 --tiered-stt-budget 2048 --sttram-bifurcation-frac 0.75 \
    --write-aware-lambda $L --max-samples 25 --json results/wa075_$L.json
  echo "===== $(date) DONE wa075_$L (exit $?) ====="
done
echo "===== $(date) ALL DONE ====="
