#!/bin/bash
# E2: fixed-sample bifurcation-frac sweep — savings x accuracy x overhead.
# All 4 points: same revision, 25 samples, TieredKV-only, vram1024/stt2048, bf16, seed 0.
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
export HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
cd /data/nishant/Nishant/Prashant/KV_Cache_Optimization
SHORT="multifieldqa_en qasper narrativeqa hotpotqa triviaqa"
for F in 1.0 0.75 0.5 0.25; do
  TAG=$(echo $F | tr '.' '_')
  echo "===== $(date) START e2_frac${TAG} ====="
  nice -n 10 python experiments/longbench_eval.py --model $MODEL --methods tieredkv \
    --tasks $SHORT --budget 1024 --seed 0 --device cuda --tiered-kv-dtype bfloat16 \
    --tiered-vram-budget 1024 --tiered-stt-budget 2048 \
    --sttram-bifurcation-frac $F --max-samples 25 \
    --json results/e2_frac${TAG}_short.json
  echo "===== $(date) DONE e2_frac${TAG} (exit $?) ====="
done
echo "===== $(date) ALL DONE ====="
