"""Collect every artefact written by the pipeline into tidy tables (CSV + Markdown)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import QA_METRICS, loo_calibrated_error, paired_bootstrap_test, proxy_error


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_training_metrics(runs_dir=config.RUNS_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(runs_dir).glob("*/training_metrics.json")):
        d = _read_json(p)
        d["run_id"] = p.parent.name
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df["method"] = df["run_id"].map(config.method_of_run)
    return df


def load_eval_metrics(eval_dir=config.EVAL_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(eval_dir).glob("*/*_metrics.json")):
        d = _read_json(p)
        d.setdefault("run_id", p.parent.name)
        d.setdefault("split", p.stem.replace("_metrics", ""))
        rows.append(d)
    df = pd.DataFrame(rows)
    if len(df):
        df["method"] = df["run_id"].map(config.method_of_run)
    return df


def load_predictions(run_id: str, split: str, eval_dir=config.EVAL_DIR) -> pd.DataFrame | None:
    p = Path(eval_dir) / run_id / f"{split}_predictions.jsonl"
    return pd.read_json(p, lines=True) if p.is_file() else None


def load_selection_summaries(selection_dir=config.SELECTION_DIR) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(selection_dir).glob("b*/selection_summary.json")):
        for r in _read_json(p).get("subsets", []):
            rows.append(r)
    return pd.DataFrame(rows)


def load_step_timings(runs_dir=config.RUNS_DIR) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(runs_dir).glob("*/step_timings.csv")):
        df = pd.read_csv(p)
        df["run_id"] = p.parent.name
        df["method"] = config.method_of_run(p.parent.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_loss_curves(runs_dir=config.RUNS_DIR) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(runs_dir).glob("*/log_history.json")):
        hist = [h for h in _read_json(p) if "loss" in h and "step" in h]
        if hist:
            df = pd.DataFrame(hist)[["step", "loss"] + (["epoch"] if "epoch" in hist[0] else [])]
            df["run_id"] = p.parent.name
            df["method"] = config.method_of_run(p.parent.name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
def build_results_table(train: pd.DataFrame, evals: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    if evals.empty:
        return pd.DataFrame()
    keep_eval = ["run_id", "method", "split", "num_examples", "generation_seconds"]
    for m in QA_METRICS:
        keep_eval += [m, f"{m}_ci_low", f"{m}_ci_high"]
    table = evals[[c for c in keep_eval if c in evals.columns]].copy()
    if not train.empty:
        cols = ["run_id", "num_train_examples", "num_synthetic_examples", "mean_seq_len", "final_train_loss",
                "predicted_training_time_sec", "actual_wall_clock_sec", "proxy_error_pct",
                "predicted_from_observed_batches_sec", "peak_gpu_memory_gb", "device_name"]
        table = table.merge(train[[c for c in cols if c in train.columns]], on="run_id", how="left")
    if not sel.empty:
        cols = ["run_id", "budget_sec", "seed", "num_selected", "predicted_epoch_time_sec", "total_utility",
                "selection_seconds", "cov_shift_rel"]
        table = table.merge(sel[[c for c in cols if c in sel.columns]], on="run_id", how="left")
    order = {m: i for i, m in enumerate(config.REFERENCE_RUNS + config.METHODS)}
    table["_o"] = table["method"].map(order).fillna(99)
    return table.sort_values(["split", "_o", "run_id"]).drop(columns="_o").reset_index(drop=True)


def seed_aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """Mean / std over seeds for each (split, method, budget)."""
    if table.empty:
        return table
    t = table.copy()
    if "budget_sec" not in t.columns:
        t["budget_sec"] = np.nan
    t["budget_sec"] = t["budget_sec"].fillna(-1)
    metrics = [m for m in QA_METRICS + ["actual_wall_clock_sec", "proxy_error_pct"] if m in t.columns]
    g = t.groupby(["split", "method", "budget_sec"])[metrics]
    agg = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    agg = agg.reset_index()
    agg["budget_sec"] = agg["budget_sec"].replace(-1, np.nan)
    return agg


def significance_vs_reference(table: pd.DataFrame, reference_method: str = "random", eval_dir=config.EVAL_DIR,
                              n_boot: int = 5000) -> pd.DataFrame:
    """Paired, passage-clustered bootstrap test of every run against the reference run with
    the same split / budget / seed (falls back to any reference run of the split)."""
    rows = []
    if table.empty:
        return pd.DataFrame()
    for split, tsplit in table.groupby("split"):
        refs = tsplit[tsplit["method"] == reference_method]
        if refs.empty:
            continue
        for _, r in tsplit.iterrows():
            if r["method"] == reference_method:
                continue
            ref = refs
            for key in ("budget_sec", "seed"):
                if key in refs.columns and pd.notna(r.get(key)):
                    narrowed = ref[ref[key] == r[key]]
                    ref = narrowed if len(narrowed) else ref
            ref_id = ref["run_id"].iloc[0]
            a = load_predictions(r["run_id"], split, eval_dir)
            b = load_predictions(ref_id, split, eval_dir)
            if a is None or b is None:
                continue
            merged = a.merge(b, on="example_id", suffixes=("_a", "_b"))
            for metric in ("token_f1", "exact_match", "char_f1"):
                res = paired_bootstrap_test(merged[f"{metric}_a"], merged[f"{metric}_b"],
                                            groups=merged["passage_id_a"], n_boot=n_boot)
                rows.append({"split": split, "run_id": r["run_id"], "method": r["method"], "reference_run": ref_id,
                             "metric": metric, "n": len(merged), "diff_pct_points": 100 * res["diff"],
                             "ci_low": 100 * res["ci_low"], "ci_high": 100 * res["ci_high"],
                             "p_value": res["p_value"], "significant_at_0.05": res["p_value"] < 0.05})
    return pd.DataFrame(rows)


def proxy_tables(train: pd.DataFrame, steps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run-level and micro-batch-level evaluation of the training-time proxy."""
    summary = {}
    run_table = pd.DataFrame()
    if not train.empty and "predicted_training_time_sec" in train.columns:
        cols = [c for c in ["run_id", "method", "num_train_examples", "predicted_training_time_sec",
                            "predicted_from_observed_batches_sec", "measured_microbatch_sum_sec",
                            "actual_wall_clock_sec", "proxy_error_pct", "padding_fraction", "device_name",
                            "proxy_profiled_on"] if c in train.columns]
        run_table = train[cols].copy()
        run_level = proxy_error(train["predicted_training_time_sec"], train["actual_wall_clock_sec"])
        run_level.update(loo_calibrated_error(train["predicted_training_time_sec"], train["actual_wall_clock_sec"]))
        summary["run_level_pre_training_estimate"] = run_level
        if "predicted_from_observed_batches_sec" in train.columns and "measured_microbatch_sum_sec" in train.columns:
            summary["run_level_observed_batches"] = proxy_error(train["predicted_from_observed_batches_sec"],
                                                                train["measured_microbatch_sum_sec"])
    if not steps.empty:
        summary["micro_batch_level"] = proxy_error(steps["proxy_predicted_ms"], steps["measured_ms"])
    return run_table, summary


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Dependency-free Markdown table."""
    if df.empty:
        return "_(no rows)_\n"

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"
