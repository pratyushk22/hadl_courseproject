"""Step 6 - draw every report figure into outputs/figures/ (CPU, seconds).

Selection-only figures work right after step 2 (no GPU needed); the model-quality and
timing figures appear once steps 3-5 have produced results.

    python scripts/06_make_plots.py [--budget_sec 600]
"""
import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from hadl_eval import config, plotting
from hadl_eval.results import load_loss_curves, load_predictions, load_step_timings, load_training_metrics
from hadl_eval.time_proxy import TimeProxy


def read_csv(path):
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget_sec", type=float, default=None, help="default: first budget of step 2")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEEDS[0], help="seed whose subsets are drawn")
    ap.add_argument("--proxy_json", default=None,
                    help="default: <outputs>/profiling/time_proxy_model.json if the run re-profiled, "
                         "else performance_profiling/time_proxy_model.json")
    args = ap.parse_args()

    if args.budget_sec is None:
        try:
            args.budget_sec = config.resolved_budgets()[0]
        except FileNotFoundError:
            args.budget_sec = 0.0
    prof_dir = config.OUTPUT_ROOT / "profiling"
    if args.proxy_json is None:
        own = prof_dir / "time_proxy_model.json"
        args.proxy_json = str(own if own.is_file() else config.TIME_PROXY_JSON)
    own_csv = prof_dir / "profiling_results.csv"
    profiling_csv = own_csv if own_csv.is_file() else config.PROFILING_CSV
    fig_dir = config.FIGURES_DIR
    res = config.RESULTS_DIR
    written = []

    # ---- selection figures (CPU only) -----------------------------------------------
    sel = read_csv(res / "selection_summary.csv")
    if sel.empty:  # aggregation not run yet -> read selection json directly
        from hadl_eval.results import load_selection_summaries
        sel = load_selection_summaries()
    written += plotting.plot_selection_overview(sel, fig_dir, args.budget_sec)

    sel_dir = config.SELECTION_DIR / f"b{config.budget_tag(args.budget_sec)}"
    subsets = {}
    for m in config.METHODS:
        p = sel_dir / f"{config.run_id_for(m, args.budget_sec, args.seed)}.parquet"
        if not p.is_file():  # seed not selected -> use any seed of this method
            p = next(iter(sorted(sel_dir.glob(f"{m}_b{config.budget_tag(args.budget_sec)}_s*.parquet"))), p)
        if p.is_file():
            subsets[m] = pd.read_parquet(p)
    written += plotting.plot_token_length_distribution(subsets, fig_dir, args.budget_sec)
    pool_path = config.POOL_DIR / "candidate_pool.parquet"
    if pool_path.is_file():
        pool = pd.read_parquet(pool_path, columns=["utility", "predicted_time_cost_ms", "cm_ranked"])
        written += plotting.plot_utility_vs_cost(pool, subsets, fig_dir, args.budget_sec)
    sweep = read_csv(config.SELECTION_DIR / "budget_sweep.csv")
    written += plotting.plot_budget_sweep(sweep, fig_dir)
    written += plotting.plot_budget_sweep(sweep, fig_dir, y="num_selected", ylabel="Examples selected")
    written += plotting.plot_covariance_shift(sweep, fig_dir)
    written += plotting.plot_proxy_profile_fit(profiling_csv, TimeProxy.from_json(args.proxy_json), fig_dir)

    # ---- training / evaluation figures (after GPU runs) ------------------------------
    table = read_csv(res / "results_table.csv")
    train = load_training_metrics()
    steps = load_step_timings()
    proxy_summary_path = res / "proxy_error_summary.json"
    proxy_summary = json.loads(proxy_summary_path.read_text()) if proxy_summary_path.is_file() else None
    written += plotting.plot_qa_scores(table, fig_dir)
    written += plotting.plot_training_time(train, fig_dir)
    written += plotting.plot_proxy_step_parity(steps, fig_dir, proxy_summary)
    written += plotting.plot_loss_curves(load_loss_curves(), fig_dir)
    written += plotting.plot_quality_vs_time(table, fig_dir)
    written += plotting.plot_metric_vs_budget(table, fig_dir)
    written += plotting.plot_pipeline_overhead(read_csv(res / "overhead_components.csv"), fig_dir)
    written += plotting.plot_covariance_diagnostic(read_csv(res / "covariance_diagnostic.csv"), fig_dir)
    if not table.empty:
        for split in table["split"].unique():
            runs = table.loc[table["split"] == split, "run_id"].unique()
            preds = {r: load_predictions(r, split) for r in runs}
            written += plotting.plot_f1_by_answer_length(preds, fig_dir, split)

    pngs = [p for p in written if p.suffix == ".png"]
    print(f"Wrote {len(pngs)} figures (PNG + PDF) to {fig_dir}:")
    for p in pngs:
        print("  ", p.name)


if __name__ == "__main__":
    main()
