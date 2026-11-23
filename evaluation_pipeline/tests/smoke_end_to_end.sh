#!/usr/bin/env bash
# End-to-end smoke test of the whole pipeline (run_pipeline.sh, steps 1-6) on CPU with a tiny,
# randomly initialised Qwen3 model (same tokenizer and chat template as Qwen3-1.7B).
# GPUS=2 exercises the parallel one-process-per-GPU path (on CPU both shards share the CPU).
# Checks that every code path runs; the scores are meaningless. ~5 minutes. Real results untouched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${SMOKE_DIR:-$(mktemp -d)}"
export HADL_OUTPUT_ROOT="$WORK/outputs"
export PYTHON="${PYTHON:-python3}"
cd "$HERE"
echo "Smoke-test outputs in $WORK"

"$PYTHON" tests/make_tiny_model.py "$WORK/tiny_qwen3"
# step 1 with a small random pool, then the runner for steps 2-6
"$PYTHON" scripts/01_build_pool.py --random_pool_size 2000 --batch_size 2
STEPS="2 3 4 5 6" GPUS="${SMOKE_GPUS:-2}" BUDGET_X_REAL=0.05 SEEDS="1 2" METHODS=all REFERENCE=real_only \
  DTYPE=fp32 BATCH=2 ACCUM=2 EVAL_BS=4 MODEL_ID="$WORK/tiny_qwen3" \
  TRAIN_EXTRA="--max_train_samples 24 --max_steps 3 --max_seq_len 512" \
  EVAL_EXTRA="--max_eval_samples 6 --max_new_tokens 8 --max_input_tokens 1024 --n_boot 200" \
  bash run_pipeline.sh
test -f "$HADL_OUTPUT_ROOT/results/results_table.csv"
test -f "$HADL_OUTPUT_ROOT/figures/qa_scores_test.png"
n_runs=$(ls -d "$HADL_OUTPUT_ROOT"/runs/*/adapter | wc -l)
n_eval=$(ls "$HADL_OUTPUT_ROOT"/eval/*/test_metrics.json | wc -l)
echo "trained runs: $n_runs, evaluated models: $n_eval"
test "$n_runs" -eq 12      # 5 methods x 2 seeds + real_only x 2 seeds
test "$n_eval" -eq $((n_runs + 1))   # + base_zero_shot
echo "SMOKE TEST PASSED"
