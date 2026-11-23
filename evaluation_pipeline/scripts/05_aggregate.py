"""Step 5 - collect all metrics into report tables (CPU, seconds).

Outputs (outputs/results/):
    results_table.csv / .md        one row per (run, split): QA metrics + CIs, sizes, times
    results_by_method.csv / .md    mean / std over seeds per (split, method, budget)
    significance_vs_random.csv/.md paired passage-clustered bootstrap vs the random baseline
    proxy_runs.csv, proxy_error_summary.json   time-proxy evaluation
    overhead_components.csv        selection-pipeline cost + training time per method
    selection_summary.csv          what every method selected
"""
import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from hadl_eval import config
from hadl_eval.results import (build_results_table, load_eval_metrics, load_selection_summaries, load_step_timings,
                               load_training_metrics, proxy_tables, seed_aggregate, significance_vs_reference,
                               to_markdown)


def overhead_components(train: pd.DataFrame, sel: pd.DataFrame, budget: float | None) -> pd.DataFrame:
    manual = json.loads(config.OVERHEAD_JSON.read_text()) if config.OVERHEAD_JSON.is_file() else {}
    pool_report_path = config.POOL_DIR / "pool_report.json"
    pool_report = json.loads(pool_report_path.read_text()) if pool_report_path.is_file() else {}
    cand_proxy = pool_report.get("candidate_pool", {}).get("proxy_scoring_seconds")
    rnd_proxy = pool_report.get("random_pool", {}).get("proxy_scoring_seconds")
    shared = {
        "Embeddings": manual.get("embedding_generation_sec"),
        "Low-rank scoring": manual.get("low_rank_selection_sec"),
        "Covariance matching": manual.get("covariance_matching_sec"),
    }
    uses = {"random": [], "low_rank_only": ["Embeddings", "Low-rank scoring"],
            "cov_match_only": ["Embeddings", "Low-rank scoring", "Covariance matching"],
            "quality_only": ["Embeddings", "Low-rank scoring", "Covariance matching"],
            "proposed_knapsack": ["Embeddings", "Low-rank scoring", "Covariance matching"]}
    rows = []
    if train.empty:
        return pd.DataFrame(rows)
    t = train if budget is None or "budget_sec" not in train.columns else \
        train[(train["budget_sec"].isna()) | np.isclose(train["budget_sec"].astype(float), budget)]
    for method, g in t.groupby("method"):
        r = g.iloc[0]
        for comp in uses.get(method, []):
            if shared.get(comp) is not None:
                rows.append({"method": method, "component": comp, "seconds": float(shared[comp])})
        proxy_sec = rnd_proxy if method == "random" else cand_proxy
        if method in uses and proxy_sec is not None:
            rows.append({"method": method, "component": "Time-proxy scoring", "seconds": float(proxy_sec)})
        if not sel.empty and r["run_id"] in set(sel["run_id"]):
            s = sel[sel["run_id"] == r["run_id"]].iloc[0]
            rows.append({"method": method, "component": "Subset selection", "seconds": float(s["selection_seconds"])})
        rows.append({"method": method, "component": "Fine-tuning", "seconds": float(r["actual_wall_clock_sec"])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget_sec", type=float, default=None,
                    help="budget used for the overhead breakdown (default: first budget of step 2)")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()

    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    train = load_training_metrics()
    evals = load_eval_metrics()
    sel = load_selection_summaries()
    steps = load_step_timings()
    print(f"Found {len(train)} training runs, {len(evals)} evaluation results, {len(sel)} selection records, "
          f"{len(steps)} timed micro-batches")

    if not sel.empty:
        sel.to_csv(out / "selection_summary.csv", index=False)

    table = build_results_table(train, evals, sel)
    cov_csv = out / "covariance_diagnostic.csv"          # written by step 7 (optional)
    if not table.empty and cov_csv.is_file():
        cov = pd.read_csv(cov_csv)
        cov = cov[cov["kind"] == "subset"][["run_id", "rel_dist_pca", "rel_dist_full"]]
        cov = cov.rename(columns={"rel_dist_pca": "cov_shift_pca", "rel_dist_full": "cov_shift_full"})
        table = table.drop(columns=[c for c in ("cov_shift_pca", "cov_shift_full") if c in table.columns]) \
                     .merge(cov, on="run_id", how="left")
    if not table.empty:
        table.to_csv(out / "results_table.csv", index=False)
        compact = ["run_id", "split", "exact_match", "token_f1", "char_f1", "contains_gold", "cov_shift_pca",
                   "num_train_examples",
                   "predicted_training_time_sec", "actual_wall_clock_sec", "proxy_error_pct"]
        (out / "results_table.md").write_text(to_markdown(table[[c for c in compact if c in table.columns]]))
        agg = seed_aggregate(table)
        agg.to_csv(out / "results_by_method.csv", index=False)
        (out / "results_by_method.md").write_text(to_markdown(agg))
        print("\n" + to_markdown(table[[c for c in compact if c in table.columns]]))

        sig = significance_vs_reference(table, "random", n_boot=args.n_boot)
        if not sig.empty:
            sig.to_csv(out / "significance_vs_random.csv", index=False)
            (out / "significance_vs_random.md").write_text(to_markdown(sig, floatfmt="{:.3f}"))
            print(to_markdown(sig[sig["metric"] == "token_f1"], floatfmt="{:.3f}"))

    run_proxy, proxy_summary = proxy_tables(train, steps)
    if not run_proxy.empty:
        run_proxy.to_csv(out / "proxy_runs.csv", index=False)
    if proxy_summary:
        (out / "proxy_error_summary.json").write_text(json.dumps(proxy_summary, indent=2))
        print("Time-proxy error:", json.dumps(proxy_summary, indent=2))

    budget = args.budget_sec
    if budget is None:
        try:
            budget = config.resolved_budgets()[0]
        except FileNotFoundError:
            budget = None
    comps = overhead_components(train, sel, budget)
    if not comps.empty:
        comps.to_csv(out / "overhead_components.csv", index=False)
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
