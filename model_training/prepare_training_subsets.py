import os
import time
import json
import argparse
import numpy as np
import pandas as pd

DEFAULT_INPUT = "../data_and_scripts/odia_synth_cleaned/or_synthetic_filtered_with_time_cost.parquet"
OUTPUT_DIR = "../data_and_scripts/odia_synth_cleaned/selected_subsets"
SUMMARY_JSON = "subset_selection_summary.json"

# Default training time budget per epoch in seconds (e.g., 600s = 10 minutes per epoch)
DEFAULT_TIME_BUDGET_SEC = 600.0
RANDOM_SEED = 42


def min_max_normalize(arr):
    arr = np.asarray(arr, dtype=np.float64)
    min_v, max_v = np.nanmin(arr), np.nanmax(arr)
    if max_v - min_v < 1e-12:
        return np.ones_like(arr) * 0.5
    return (arr - min_v) / (max_v - min_v)


def ensure_utility_scores(df, seed=RANDOM_SEED):
    """
    Uses precomputed Low-Rank (Aditya Jain) and Covariance Matching (Vanshika) columns
    if present in the dataframe; otherwise synthesizes deterministic proxy utility scores
    so the subset selection and training pipeline can be tested standalone.
    """
    rng = np.random.default_rng(seed)

    # 1. Detect or synthesize Low-Rank Residual score
    lr_candidates = ["low_rank_score", "residual_score", "lr_score"]
    lr_col = next((c for c in lr_candidates if c in df.columns), None)
    if lr_col is not None:
        print(f"Found existing Low-Rank score column: '{lr_col}'")
        lr_norm = min_max_normalize(df[lr_col].values)
        score_source_lr = lr_col
    else:
        print("Low-Rank column not found yet -> generating deterministic proxy low_rank_score for standalone run.")
        # Slightly favor informative medium-length passages with controlled variance
        tok_len = df["token_length"].values if "token_length" in df.columns else np.full(len(df), 200.0)
        raw_lr = np.log1p(tok_len) + rng.normal(0, 0.35, size=len(df))
        lr_norm = min_max_normalize(raw_lr)
        df["low_rank_score"] = np.round(lr_norm, 6)
        score_source_lr = "synthetic_fallback"

    # 2. Detect or synthesize Covariance Matching score
    cov_candidates = ["cov_match_score", "covariance_score", "cov_score"]
    cov_col = next((c for c in cov_candidates if c in df.columns), None)
    if cov_col is not None:
        print(f"Found existing Covariance Matching score column: '{cov_col}'")
        cov_norm = min_max_normalize(df[cov_col].values)
        score_source_cov = cov_col
    else:
        print("Covariance Matching column not found yet -> generating deterministic proxy cov_match_score for standalone run.")
        raw_cov = rng.beta(a=2.5, b=2.0, size=len(df))
        cov_norm = min_max_normalize(raw_cov)
        df["cov_match_score"] = np.round(cov_norm, 6)
        score_source_cov = "synthetic_fallback"

    # Combined usefulness score u_i (equal weighting of normalized Low-Rank + Covariance Matching)
    df["combined_utility_score"] = np.round(0.5 * lr_norm + 0.5 * cov_norm, 6)
    return df, score_source_lr, score_source_cov


def select_within_budget(df, order_indices, budget_ms):
    """
    Greedily selects examples according to `order_indices` until cumulative
    `predicted_time_cost_ms` reaches `budget_ms`.
    """
    costs = df["predicted_time_cost_ms"].values
    ordered_costs = costs[order_indices]
    cum_costs = np.cumsum(ordered_costs)
    valid_mask = cum_costs <= budget_ms
    selected_indices = order_indices[valid_mask]
    return df.iloc[selected_indices].copy()


def main():
    parser = argparse.ArgumentParser(description="Select training subsets for the 3 HADL baselines within a fixed time budget.")
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--budget_sec", type=float, default=DEFAULT_TIME_BUDGET_SEC,
                        help="Allocated training time budget per epoch in seconds.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    budget_ms = args.budget_sec * 1000.0

    print(f"Loading cost-annotated dataset from {args.input_file}...")
    df = pd.read_parquet(args.input_file)
    assert "predicted_time_cost_ms" in df.columns, "Missing 'predicted_time_cost_ms'! Run Yatika's assign_time_costs.py first."

    df, src_lr, src_cov = ensure_utility_scores(df)

    summary = {
        "input_file": args.input_file,
        "total_pool_examples": int(len(df)),
        "time_budget_sec_per_epoch": args.budget_sec,
        "low_rank_score_source": src_lr,
        "covariance_score_source": src_cov,
        "methods": {}
    }

    # ------------------------------------------------------------------
    # Baseline 1: Random Selection within Time Budget
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)
    rand_indices = rng.permutation(len(df))
    df_random = select_within_budget(df, rand_indices, budget_ms)
    t_random_sec = time.perf_counter() - t0

    random_path = os.path.join(args.output_dir, "subset_random.parquet")
    df_random.to_parquet(random_path, index=False)

    # ------------------------------------------------------------------
    # Baseline 2: Two Methods Combined (Quality-Only, Ignoring Time Cost)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    quality_indices = np.argsort(-df["combined_utility_score"].values, kind="mergesort")
    df_quality = select_within_budget(df, quality_indices, budget_ms)
    t_quality_sec = time.perf_counter() - t0

    quality_path = os.path.join(args.output_dir, "subset_quality_only.parquet")
    df_quality.to_parquet(quality_path, index=False)

    # ------------------------------------------------------------------
    # Baseline 3 (Proposed Method): Knapsack (Two Methods + Time Proxy)
    # Maximize sum(u_i) subject to sum(c_i) <= Budget via utility density u_i / c_i
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    safe_cost_ms = np.maximum(df["predicted_time_cost_ms"].values, 1e-6)
    df["knapsack_density_score"] = df["combined_utility_score"].values / safe_cost_ms
    knapsack_indices = np.argsort(-df["knapsack_density_score"].values, kind="mergesort")
    df_knapsack = select_within_budget(df, knapsack_indices, budget_ms)
    t_knapsack_sec = time.perf_counter() - t0

    knapsack_path = os.path.join(args.output_dir, "subset_proposed_knapsack.parquet")
    df_knapsack.to_parquet(knapsack_path, index=False)

    # ------------------------------------------------------------------
    # Log & Save Summary Comparison
    # ------------------------------------------------------------------
    print("\n=== Subset Selection Summary (Budget = {:.1f}s per epoch) ===".format(args.budget_sec))
    print(f"{'Method':<24} | {'Samples':>8} | {'Avg Tokens':>10} | {'Total Utility':>14} | {'Pred Time (s)':>13} | {'Select Time (s)':>15}")
    print("-" * 98)

    for name, sub_df, sel_sec, path in [
        ("1. Random Selection", df_random, t_random_sec, random_path),
        ("2. Quality-Only (LR+Cov)", df_quality, t_quality_sec, quality_path),
        ("3. Proposed Knapsack", df_knapsack, t_knapsack_sec, knapsack_path),
    ]:
        n_samples = len(sub_df)
        avg_tok = float(sub_df["token_length"].mean()) if n_samples > 0 else 0.0
        tot_util = float(sub_df["combined_utility_score"].sum()) if n_samples > 0 else 0.0
        pred_sec = float(sub_df["predicted_time_cost_ms"].sum() / 1000.0) if n_samples > 0 else 0.0

        print(f"{name:<24} | {n_samples:>8} | {avg_tok:>10.1f} | {tot_util:>14.2f} | {pred_sec:>13.2f} | {sel_sec:>15.4f}")

        key = name.split(".")[1].strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        summary["methods"][key] = {
            "output_file": path,
            "num_selected_samples": int(n_samples),
            "avg_token_length": round(avg_tok, 2),
            "total_combined_utility": round(tot_util, 4),
            "predicted_epoch_time_sec": round(pred_sec, 3),
            "selection_algorithm_time_sec": round(sel_sec, 5)
        }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved selection metrics to '{SUMMARY_JSON}' and subsets to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
