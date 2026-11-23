#!/usr/bin/env bash
# Full HADL evaluation pipeline. Run from anywhere:
#
#   bash evaluation_pipeline/run_pipeline.sh                 # everything, default settings
#   PROFILE=1 bash evaluation_pipeline/run_pipeline.sh       # first re-profile the time proxy on THIS GPU
#   SEEDS="42 43 44" BUDGET_X_REAL="0.5 1 2" METHODS=all REFERENCE="real_only" bash evaluation_pipeline/run_pipeline.sh
#   STEPS="5 6" bash evaluation_pipeline/run_pipeline.sh     # only aggregate + plot
#
#   Kaggle 2xT4:
#   PROFILE=1 GPUS=2 BATCH=2 ACCUM=8 EVAL_BS=4 REFERENCE=real_only bash evaluation_pipeline/run_pipeline.sh
#
# Environment variables (all optional):
#   STEPS          which steps to run (default "0 1 2 3 4 5 6 7"; step 0 runs only with PROFILE=1).
#                  Order of execution: 0 1 2 3 4 7 5 6 (the covariance diagnostic, 7, runs before
#                  the tables and plots so they include it; it is optional and never fails the run)
#   PROFILE=1      re-profile Qwen3-1.7B LoRA step time on the local GPU and use that proxy
#   PROXY_JSON     time-proxy json to use (default: performance_profiling/time_proxy_model.json)
#   BUDGET_X_REAL  synthetic budget(s) as multiples of the real-train proxy time (default "1.0")
#   BUDGET_SEC     absolute budget(s) in proxy seconds (overrides BUDGET_X_REAL)
#   SEEDS          training / random-selection seeds (default "42")
#   METHODS        methods to train: "random quality_only proposed_knapsack" (default) or "all"
#   REFERENCE      extra reference runs, e.g. "real_only"
#   DTYPE          auto | bf16 | fp16 | fp32 (default auto: bf16 on A100/L4, fp16 on T4/P100)
#   EPOCHS         training epochs (default 1)
#   BATCH, ACCUM   per-GPU micro-batch and gradient accumulation (default 4 x 4; use 2 x 8 on 16 GB GPUs)
#   EVAL_BS        evaluation batch size (default 8; use 4 on a 16 GB GPU)
#   GPUS           number of GPUs to use in parallel: the job list is split, one single-GPU
#                  process per GPU (default 1; use 2 on Kaggle 2xT4). Each run is still timed on one GPU.
#   MODEL_ID       base model (default Qwen/Qwen3-1.7B)
#   TRAIN_EXTRA, EVAL_EXTRA   extra arguments for scripts/03_train.py / 04_evaluate.py (debugging)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PYTHON:-python}"
STEPS="${STEPS:-0 1 2 3 4 5 6 7}"
SEEDS="${SEEDS:-42}"
METHODS="${METHODS:-random quality_only proposed_knapsack}"
REFERENCE="${REFERENCE:-}"
DTYPE="${DTYPE:-auto}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-4}"
ACCUM="${ACCUM:-4}"
EVAL_BS="${EVAL_BS:-8}"
GPUS="${GPUS:-1}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-1.7B}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
EVAL_EXTRA="${EVAL_EXTRA:-}"
BUDGET_X_REAL="${BUDGET_X_REAL:-1.0}"
USER_PROXY="${PROXY_JSON:-}"
PROXY_JSON="${PROXY_JSON:-$ROOT/performance_profiling/time_proxy_model.json}"
OUT="${HADL_OUTPUT_ROOT:-$HERE/outputs}"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

# Run "$@" once per GPU with --shard i/GPUS and CUDA_VISIBLE_DEVICES=i; fail if any shard fails.
run_sharded() {
  local name="$1"; shift
  if [[ "$GPUS" -le 1 ]]; then
    "$@"
    return
  fi
  local pids=() i
  for ((i = 0; i < GPUS; i++)); do
    echo "  shard $i/$GPUS on GPU $i -> log: $LOG_DIR/${name}_shard$i.log"
    CUDA_VISIBLE_DEVICES=$i "$@" --shard "$i/$GPUS" > "$LOG_DIR/${name}_shard$i.log" 2>&1 &
    pids+=($!)
  done
  local failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "!! shard $i of $name FAILED - last lines of its log:"
      tail -n 30 "$LOG_DIR/${name}_shard$i.log" || true
      failed=1
    else
      tail -n 3 "$LOG_DIR/${name}_shard$i.log" || true
    fi
  done
  return $failed
}

cd "$HERE"

if has_step 0 && [[ "${PROFILE:-0}" == "1" ]]; then
  echo "=== Step 0: profile LoRA step time on this GPU and fit the time proxy"
  PROF_DIR="$OUT/profiling"
  mkdir -p "$PROF_DIR"
  
  (cd "$PROF_DIR" && CUDA_VISIBLE_DEVICES=0 "$PY" - "$ROOT/performance_profiling/profile_hardware.py" <<'PYEOF'
import runpy, sys
import torch
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
    torch.cuda.is_bf16_supported = lambda *args, **kwargs: False
sys.dont_write_bytecode = True
runpy.run_path(sys.argv[1], run_name="__main__")
PYEOF
  ) && (cd "$PROF_DIR" && "$PY" "$ROOT/performance_profiling/fit_time_proxy.py")
  PROXY_JSON="$PROF_DIR/time_proxy_model.json"
  export HADL_PROFILING_CSV="$PROF_DIR/profiling_results.csv"
  echo "Using freshly fitted proxy: $PROXY_JSON"
elif [[ -z "$USER_PROXY" && -f "$OUT/profiling/time_proxy_model.json" ]]; then
  PROXY_JSON="$OUT/profiling/time_proxy_model.json"
  export HADL_PROFILING_CSV="$OUT/profiling/profiling_results.csv"
  echo "Re-using proxy profiled earlier on this machine: $PROXY_JSON"
fi

if has_step 1; then
  echo "=== Step 1: build the costed candidate pool (cost assumes micro-batch $BATCH)"
  "$PY" scripts/01_build_pool.py --proxy_json "$PROXY_JSON" --batch_size "$BATCH"
fi

if has_step 2; then
  echo "=== Step 2: select subsets"
  if [[ -n "${BUDGET_SEC:-}" ]]; then
    "$PY" scripts/02_select_subsets.py --budget_sec $BUDGET_SEC --seeds $SEEDS --sweep
  else
    "$PY" scripts/02_select_subsets.py --budget_x_real $BUDGET_X_REAL --seeds $SEEDS --sweep
  fi
fi

if has_step 3; then
  echo "=== Step 3: fine-tune Qwen3-1.7B + LoRA on $GPUS GPU(s)"
  REF_ARGS=()
  [[ -n "$REFERENCE" ]] && REF_ARGS=(--reference_runs $REFERENCE)
  run_sharded train "$PY" scripts/03_train.py --methods $METHODS ${REF_ARGS[@]+"${REF_ARGS[@]}"} --seeds $SEEDS \
        --dtype "$DTYPE" --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" --proxy_json "$PROXY_JSON" \
        --model_id "$MODEL_ID" $TRAIN_EXTRA
fi

if has_step 4; then
  echo "=== Step 4: evaluate on the real validation + test splits on $GPUS GPU(s)"
  run_sharded eval "$PY" scripts/04_evaluate.py --splits validation test --dtype "$DTYPE" --batch_size "$EVAL_BS" \
        --model_id "$MODEL_ID" $EVAL_EXTRA
fi

if has_step 7; then
  echo "=== Step 7: covariance-shift diagnostic (embeddings from the team Drive; optional)"
  if ! CUDA_VISIBLE_DEVICES=0 "$PY" scripts/07_covariance_diagnostic.py --outputs "$OUT" \
        > "$LOG_DIR/covariance_diagnostic.log" 2>&1; then
    echo "!! covariance diagnostic failed (optional) - continuing. Last lines of $LOG_DIR/covariance_diagnostic.log:"
    tail -n 15 "$LOG_DIR/covariance_diagnostic.log" || true
  else
    grep -E "^\| |Wrote|parity|\[drive\]" "$LOG_DIR/covariance_diagnostic.log" || true
  fi
fi

if has_step 5; then
  echo "=== Step 5: aggregate results"
  "$PY" scripts/05_aggregate.py
fi

if has_step 6; then
  echo "=== Step 6: plots"
  "$PY" scripts/06_make_plots.py --proxy_json "$PROXY_JSON"
fi

echo "Done. Tables: $OUT/results   Figures: $OUT/figures   Logs: $LOG_DIR"
