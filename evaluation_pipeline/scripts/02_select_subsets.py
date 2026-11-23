"""Step 2 - select one training subset per method / budget / seed.

Outputs: outputs/selections/b<budget>/<run_id>.parquet
         outputs/selections/b<budget>/selection_summary.json
         outputs/selections/budget_sweep.csv          (with --sweep)

CPU only, seconds.

    python scripts/02_select_subsets.py                      # budget = 1.0 x real-train proxy time
    python scripts/02_select_subsets.py --budget_x_real 0.5 1 2 --seeds 42 43 44
    python scripts/02_select_subsets.py --budget_sec 600     # absolute proxy seconds instead
    python scripts/02_select_subsets.py --sweep              # + selection-only budget sweep
"""
import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from hadl_eval import config
from hadl_eval.diagnostics import CovarianceDiagnostic
from hadl_eval.selection import run_selection, subset_stats


def select_for_budget(pool, random_pool, budget_sec, seeds, methods, candidate_set, solver, resolution, diag,
                      write=True):
    out_dir = config.SELECTION_DIR / f"b{config.budget_tag(budget_sec)}"
    rows = []
    for method in methods:
        # deterministic methods do not depend on the seed -> select once, reuse for every seed
        for seed in (seeds if method == "random" else seeds[:1]):
            kwargs = {}
            if method in ("quality_only", "proposed_knapsack"):
                kwargs["candidate_set"] = candidate_set
            if method == "proposed_knapsack":
                kwargs.update(solver=solver, resolution=resolution)
            sub, info = run_selection(method, pool, random_pool, budget_sec * 1000.0, seed, **kwargs)
            stats = subset_stats(sub, budget_sec * 1000.0)
            if diag is not None:
                stats["cov_shift_rel"] = diag.relative_distance(sub["_source_row_idx"].values)
            for s in (seeds if method != "random" else [seed]):
                run_id = config.run_id_for(method, budget_sec, s)
                row = {"run_id": run_id, "method": method, "budget_sec": budget_sec, "seed": s, **stats, **info}
                rows.append(row)
                if write:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    sub.to_parquet(out_dir / f"{run_id}.parquet", index=False)
    if write:
        (out_dir / "selection_summary.json").write_text(json.dumps({"budget_sec": budget_sec, "subsets": rows},
                                                                    indent=2, default=float))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget_sec", type=float, nargs="+", default=None,
                    help="absolute synthetic training-time budget(s) per epoch in proxy seconds")
    ap.add_argument("--budget_x_real", type=float, nargs="+", default=config.DEFAULT_BUDGET_X_REAL,
                    help="budget(s) as multiples of the proxy-predicted epoch time of the real train split "
                         "(used when --budget_sec is not given)")
    ap.add_argument("--seeds", type=int, nargs="+", default=config.DEFAULT_SEEDS)
    ap.add_argument("--methods", nargs="+", default=config.METHODS, choices=config.METHODS)
    ap.add_argument("--candidate_set", default="cm_ranked", choices=["cm_ranked", "lr_pool"],
                    help="candidates for quality_only / proposed_knapsack")
    ap.add_argument("--solver", default="auto", choices=["auto", "dp", "greedy"])
    ap.add_argument("--resolution", type=int, default=2000, help="DP budget discretisation")
    ap.add_argument("--sweep", action="store_true", help="also run (without saving subsets) the budget sweep")
    args = ap.parse_args()

    pool = pd.read_parquet(config.POOL_DIR / "candidate_pool.parquet")
    random_pool = pd.read_parquet(config.POOL_DIR / "random_pool.parquet")
    real_cost_sec = float(pd.read_parquet(config.POOL_DIR / "real_train.parquet",
                                          columns=["predicted_time_cost_ms"])["predicted_time_cost_ms"].sum() / 1000)
    if args.budget_sec:
        budgets = [float(b) for b in args.budget_sec]
    else:
        budgets = [float(round(x * real_cost_sec)) for x in args.budget_x_real]
    print(f"Real-train proxy epoch time: {real_cost_sec:.0f} s -> budgets {budgets} s")
    config.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    (config.SELECTION_DIR / config.BUDGETS_JSON_NAME).write_text(json.dumps(
        {"budgets_sec": budgets, "real_train_proxy_sec": real_cost_sec,
         "budget_x_real": [b / real_cost_sec for b in budgets]}, indent=2))
    diag = CovarianceDiagnostic.try_create()
    print("Covariance diagnostic in step 2:", "enabled" if diag else
          "skipped (embedding .pt files not present yet; step 7 fetches them and runs it)")

    table = []
    for b in budgets:
        table += select_for_budget(pool, random_pool, b, args.seeds, args.methods, args.candidate_set,
                                   args.solver, args.resolution, diag)
    df = pd.DataFrame(table)
    show = ["run_id", "num_selected", "predicted_epoch_time_sec", "mean_token_length", "total_utility",
            "selection_seconds"] + (["cov_shift_rel"] if "cov_shift_rel" in df.columns else [])
    print(df[[c for c in show if c in df.columns]].to_string(index=False))

    if args.sweep:
        sweep = []
        for b in sorted({float(round(x * real_cost_sec)) for x in config.BUDGET_SWEEP_X_REAL} | set(budgets)):
            sweep += select_for_budget(pool, random_pool, b, args.seeds, args.methods, args.candidate_set,
                                       args.solver, args.resolution, diag, write=False)
        config.SELECTION_DIR.mkdir(parents=True, exist_ok=True)
        sweep_df = pd.DataFrame(sweep)
        sweep_df["budget_x_real"] = sweep_df["budget_sec"] / real_cost_sec
        sweep_df.to_csv(config.SELECTION_DIR / "budget_sweep.csv", index=False)
        print(f"Budget sweep written to {config.SELECTION_DIR / 'budget_sweep.csv'}")


if __name__ == "__main__":
    main()
