"""Step 1 - integrate the teammates' outputs into one costed candidate pool.

Inputs : low-rank draws (Aditya Jain), covariance-matching ranking (Vanshika),
         time proxy (Yatika), real splits (Pratyush)
Outputs: outputs/pool/candidate_pool.parquet   unique LR rows + lr/cm scores + utility + cost
         outputs/pool/random_pool.parquet      shuffled slice of the full synthetic pool + cost
         outputs/pool/real_train.parquet       real IndicQA train rows + cost
         outputs/pool/pool_report.json         sizes, timings, contamination counts

CPU only (needs the Qwen3 tokenizer; ~2-4 minutes).

    python scripts/01_build_pool.py [--alpha 0.5] [--proxy_json ...] [--max_seq_len 2048]
"""
import argparse
import json
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from hadl_eval import config
from hadl_eval.data import load_real_split
from hadl_eval.pool import (build_candidate_pool, build_random_pool, contamination_flags, eval_shingles)
from hadl_eval.prompts import load_tokenizer
from hadl_eval.time_proxy import TimeProxy, annotate_costs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=float, default=config.DEFAULT_ALPHA,
                    help="weight of the low-rank score in u_i = a*lr_pct + (1-a)*cm_score")
    ap.add_argument("--proxy_json", default=str(config.TIME_PROXY_JSON))
    ap.add_argument("--max_seq_len", type=int, default=config.MAX_SEQ_LEN)
    ap.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="training batch size the cost assumes")
    ap.add_argument("--random_pool_size", type=int, default=config.RANDOM_POOL_SIZE)
    ap.add_argument("--keep_contaminated", action="store_true",
                    help="keep synthetic rows sharing an 8-word shingle with a real val/test passage")
    ap.add_argument("--tokenizer", default=config.MODEL_ID)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config.POOL_DIR.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(args)}

    tokenizer = load_tokenizer(args.tokenizer)
    proxy = TimeProxy.from_json(args.proxy_json, max_seq_len=args.max_seq_len)
    print(f"Time proxy: {proxy.source} (profiled on {proxy.gpu_name})")

    # ---- candidate pool (low-rank x covariance matching) --------------------------------
    pool, info = build_candidate_pool(alpha=args.alpha)
    report["candidate_pool"] = info
    print(f"Candidate pool: {info['n_unique_candidates']:,} unique LR rows, {info['n_cm_ranked']:,} CM-ranked")

    t0 = time.perf_counter()
    shingles = eval_shingles()
    pool["overlaps_eval"] = contamination_flags(pool["passage"], shingles)
    report["contamination_check_seconds"] = time.perf_counter() - t0
    report["candidate_pool"]["n_overlapping_eval"] = int(pool["overlaps_eval"].sum())
    if not args.keep_contaminated and pool["overlaps_eval"].any():
        print(f"Dropping {int(pool['overlaps_eval'].sum())} candidate rows that overlap real val/test passages")
        pool = pool[~pool["overlaps_eval"]].reset_index(drop=True)
    pool = pool[pool["answer"].str.len() > 0].reset_index(drop=True)

    pool, sec = annotate_costs(pool, tokenizer, proxy, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                               desc="candidate token lengths")
    report["candidate_pool"]["proxy_scoring_seconds"] = sec
    report["candidate_pool"]["proxy_scoring_ms_per_example"] = 1000 * sec / max(len(pool), 1)
    report["candidate_pool"]["n_final"] = int(len(pool))
    report["candidate_pool"]["total_predicted_sec"] = float(pool["predicted_time_cost_ms"].sum() / 1000)
    pool.to_parquet(config.POOL_DIR / "candidate_pool.parquet", index=False)

    # ---- random-baseline pool (full synthetic train pool) -------------------------------
    rnd = build_random_pool(args.random_pool_size, seed=args.seed)
    rnd["overlaps_eval"] = contamination_flags(rnd["passage"], shingles)
    report["random_pool"] = {"source": str(config.LR_TRAIN_POOL), "n_sampled": int(len(rnd)),
                             "n_overlapping_eval": int(rnd["overlaps_eval"].sum())}
    if not args.keep_contaminated:
        rnd = rnd[~rnd["overlaps_eval"]]
    rnd = rnd[rnd["answer"].str.len() > 0].reset_index(drop=True)
    # Put the random rows on the same utility scale (for diagnostics only; random ignores it).
    audit_path = config.PROJECT_ROOT / "low_rank_method" / "residual_computations" / "train_selection_audit.parquet"
    if audit_path.is_file():
        audit = pd.read_parquet(audit_path, columns=["_source_row_idx", "sampling_probability"])
        rnd = rnd.merge(audit.rename(columns={"sampling_probability": "lr_score"}), on="_source_row_idx", how="left")
        ref = np.sort(pool["lr_score"].values)
        rnd["lr_pct"] = np.searchsorted(ref, rnd["lr_score"].fillna(0).values, side="right") / len(ref)
        cm_map = pool.set_index("_source_row_idx")["cm_score"]
        rnd["cm_score"] = rnd["_source_row_idx"].map(cm_map).fillna(0.0)
        rnd["utility"] = args.alpha * rnd["lr_pct"] + (1 - args.alpha) * rnd["cm_score"]
    rnd, sec = annotate_costs(rnd, tokenizer, proxy, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                              desc="random-pool token lengths")
    report["random_pool"]["proxy_scoring_seconds"] = sec
    report["random_pool"]["n_final"] = int(len(rnd))
    report["random_pool"]["total_predicted_sec"] = float(rnd["predicted_time_cost_ms"].sum() / 1000)
    rnd.to_parquet(config.POOL_DIR / "random_pool.parquet", index=False)

    # ---- real training data (mixed into every fine-tuning run) ---------------------------
    real = load_real_split("train")
    real, sec = annotate_costs(real, tokenizer, proxy, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
                               desc="real-train token lengths")
    real.to_parquet(config.POOL_DIR / "real_train.parquet", index=False)
    report["real_train"] = {"n": int(len(real)), "total_predicted_sec": float(real["predicted_time_cost_ms"].sum() / 1000)}

    (config.POOL_DIR / "pool_report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "args"}, indent=2, default=float))
    print(f"Wrote pool files to {config.POOL_DIR}")


if __name__ == "__main__":
    main()
