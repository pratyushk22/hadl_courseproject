"""Step 3 - fine-tune Qwen3-1.7B + LoRA (r=8) on every selected subset (GPU).

Each run trains on  real IndicQA train (886 rows, same for every method)  +  the method's
synthetic subset. The real rows are the "real data in the mix" that the covariance-matching
theory assumes; use --no_real to train on the synthetic subset alone.

Outputs per run: outputs/runs/<run_id>/{adapter/, training_metrics.json, log_history.json,
                                         step_timings.csv}

    python scripts/03_train.py --seeds 42                                   # headline methods
    python scripts/03_train.py --methods all --reference_runs real_only    # + ablations / references
    # two GPUs, one job list split across them (recommended on Kaggle 2xT4):
    CUDA_VISIBLE_DEVICES=0 python scripts/03_train.py --shard 0/2 &  CUDA_VISIBLE_DEVICES=1 python scripts/03_train.py --shard 1/2
"""
import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from hadl_eval import config
from hadl_eval.data import training_frame
from hadl_eval.modeling import is_main_process
from hadl_eval.training import train_lora


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=config.HEADLINE_METHODS,
                    help=f"subset methods to train, or 'all' ({config.METHODS})")
    ap.add_argument("--reference_runs", nargs="*", default=[], choices=["real_only"],
                    help="extra reference runs (real_only = fine-tune on the 886 real rows only)")
    ap.add_argument("--budget_sec", type=float, nargs="+", default=None,
                    help="default: the budgets chosen in step 2 (outputs/selections/budgets.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    ap.add_argument("--no_real", action="store_true", help="do not mix the real train rows into training")
    ap.add_argument("--model_id", default=config.MODEL_ID)
    ap.add_argument("--tokenizer_id", default=None, help="defaults to --model_id")
    ap.add_argument("--epochs", type=float, default=config.EPOCHS)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM_STEPS)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--max_seq_len", type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                    help="auto = bf16 on A100, fp16 on T4; use fp32 if fp16 gives NaN losses")
    ap.add_argument("--max_steps", type=int, default=-1, help="debug: stop after N optimizer steps")
    ap.add_argument("--max_train_samples", type=int, default=-1, help="debug: cap the number of training rows")
    ap.add_argument("--proxy_json", default=str(config.TIME_PROXY_JSON))
    ap.add_argument("--no_step_timing", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="retrain runs that already have results")
    ap.add_argument("--shard", default=None,
                    help="'i/n': run only every n-th job starting at i (one process per GPU, e.g. on Kaggle 2xT4)")
    args = ap.parse_args()

    methods = config.METHODS if args.methods == ["all"] else args.methods
    unknown = set(methods) - set(config.METHODS)
    if unknown:
        raise SystemExit(f"unknown methods {unknown}")

    real = training_frame(pd.read_parquet(config.POOL_DIR / "real_train.parquet"))
    jobs = []
    for seed in args.seeds:
        if "real_only" in args.reference_runs:
            jobs.append((config.run_id_for("real_only", None, seed), "real_only", None, seed))
        for b in config.resolved_budgets(args.budget_sec):
            for m in methods:
                jobs.append((config.run_id_for(m, b, seed), m, b, seed))

    shard_i, shard_n = config.parse_shard(args.shard)
    jobs = jobs[shard_i::shard_n]
    print(f"{len(jobs)} training job(s) for shard {shard_i}/{shard_n}: {[j[0] for j in jobs]}")
    summaries = []
    for run_id, method, budget, seed in jobs:
        out_dir = config.RUNS_DIR / run_id
        if (out_dir / "training_metrics.json").is_file() and not args.overwrite:
            print(f"[skip] {run_id} already trained (use --overwrite to redo)")
            continue
        if method == "real_only":
            train_df = real.copy()
        else:
            sel_path = config.SELECTION_DIR / f"b{config.budget_tag(budget)}" / f"{run_id}.parquet"
            if not sel_path.is_file():
                raise FileNotFoundError(f"{sel_path} missing - run scripts/02_select_subsets.py first")
            syn = training_frame(pd.read_parquet(sel_path))
            train_df = syn if args.no_real else pd.concat([real, syn], ignore_index=True)
        if args.max_train_samples > 0 and len(train_df) > args.max_train_samples:
            train_df = train_df.sample(n=args.max_train_samples, random_state=seed).reset_index(drop=True)
        print(f"\n=== {run_id}: {len(train_df)} rows "
              f"({int((train_df['source'] == 'synthetic').sum())} synthetic + {int((train_df['source'] == 'real').sum())} real)")
        metrics = train_lora(
            train_df, out_dir, model_id=args.model_id, tokenizer_id=args.tokenizer_id, epochs=args.epochs,
            batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr, max_seq_len=args.max_seq_len,
            seed=seed, dtype_name=args.dtype, max_steps=args.max_steps, time_steps=not args.no_step_timing,
            proxy_json=args.proxy_json,
            extra_metadata={"run_id": run_id, "method": method, "budget_sec": budget, "include_real": not args.no_real},
        )
        if is_main_process():
            print(json.dumps({k: metrics.get(k) for k in ("final_train_loss", "predicted_training_time_sec",
                                                         "actual_wall_clock_sec", "proxy_error_pct")}, indent=2))
        summaries.append(metrics)

    if is_main_process() and summaries:
        config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (config.RUNS_DIR / f"last_training_batch_summary_shard{shard_i}.json").write_text(json.dumps(summaries, indent=2, default=str))


if __name__ == "__main__":
    main()
