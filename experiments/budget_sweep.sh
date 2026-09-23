#!/usr/bin/env bash
# budget_sweep.sh — Run longbench_eval at multiple KV budgets for all methods.
# Produces results/budget_sweep/budget_<N>.json for each budget.
# This gives the "F1 vs KV Cache Budget" curves like Quest/H2O/SnapKV papers.

set -e
PYTHON=/data/nishant/Nishant/Prashant/Project/venv_kvcache/bin/python
MODEL=/data/nishant/Nishant/Prashant/Project/models/mistral-7b-instruct
HF_HOME=/data/nishant/Nishant/Prashant/Project/hf_cache
export HF_HOME

METHODS="full streamingllm h2o snapkv tieredkv"
TASKS="multifieldqa_en hotpotqa triviaqa qasper narrativeqa gov_report"
MAX_SAMPLES=25   # fast validation; change to None for full run

mkdir -p results/budget_sweep

for BUDGET in 256 512 1024 2048 4096; do
    OUTFILE="results/budget_sweep/budget_${BUDGET}.json"
    if [ -f "$OUTFILE" ]; then
        echo "[SKIP] budget=$BUDGET already done ($OUTFILE)"
        continue
    fi
    echo ""
    echo "=============================="
    echo " Running budget=$BUDGET"
    echo "=============================="
    $PYTHON experiments/longbench_eval.py \
        --model "$MODEL" \
        --budget "$BUDGET" \
        --methods $METHODS \
        --tasks $TASKS \
        --max-samples "$MAX_SAMPLES" \
        --json "$OUTFILE" \
        --device cuda \
        2>&1 | tee "results/budget_sweep/budget_${BUDGET}.log"
    echo "Done budget=$BUDGET → $OUTFILE"
done

echo ""
echo "All budget sweeps complete."
