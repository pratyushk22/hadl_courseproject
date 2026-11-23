"""All report figures (matplotlib, PNG at 200 dpi + vector PDF).

Design rules applied throughout: one y-axis per chart (no dual axes), a fixed colour per
method (config.METHOD_COLORS - colour follows the method, never its rank), thin marks,
recessive grid, legend whenever there are >= 2 series, and text in neutral ink.
Every function returns the list of written files and silently skips when its data is empty.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
POOL_GREY = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "lines.linewidth": 2.0, "font.size": 9,
})

METRIC_LABELS = {"exact_match": "Exact match (%)", "token_f1": "Token F1 (%)", "char_f1": "Character F1 (%)",
                 "contains_gold": "Answer contained (%)"}


def color(method: str) -> str:
    return config.METHOD_COLORS.get(method, INK_2)


def label(method: str) -> str:
    return config.METHOD_LABELS.get(method, method)


def method_order(methods) -> list[str]:
    known = config.REFERENCE_RUNS + config.METHODS
    present = list(dict.fromkeys(methods))
    return [m for m in known if m in present] + [m for m in present if m not in known]


def _log_budget_axis(ax, ticks, relative: bool):
    """Log x-axis with exactly one label per budget (no overlapping minor-tick labels)."""
    from matplotlib.ticker import NullFormatter, NullLocator

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(ticks, [f"{v:g}×" if relative else f"{v:,.0f}" for v in ticks])
    ax.set_xlabel("Synthetic budget per epoch, × proxy time of the real train split (log scale)" if relative
                  else "Proxy training-time budget per epoch (s, log scale)")


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------------------
# 1. QA quality per method with 95% CIs
# --------------------------------------------------------------------------------------
def plot_qa_scores(table: pd.DataFrame, out_dir: Path, metrics=("exact_match", "token_f1", "char_f1")) -> list[Path]:
    if table.empty:
        return []
    written = []
    for split, t in table.groupby("split"):
        # mean over seeds per method (CI of the first seed's bootstrap is kept as uncertainty)
        g = t.groupby("method")
        methods = method_order(t["method"])
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 0.55 * len(methods) + 1.6),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, metrics):
            if m not in t.columns:
                ax.set_visible(False)
                continue
            means = g[m].mean().reindex(methods)
            lo = g[f"{m}_ci_low"].mean().reindex(methods)
            hi = g[f"{m}_ci_high"].mean().reindex(methods)
            y = np.arange(len(methods))[::-1]
            ax.barh(y, means.values, height=0.62, color=[color(x) for x in methods], edgecolor=SURFACE, linewidth=2)
            ax.errorbar(means.values, y, xerr=[means.values - lo.values, hi.values - means.values], fmt="none",
                        ecolor=INK, elinewidth=1.2, capsize=3)
            xmax = max(5.0, float(np.nanmax(hi.values)) * 1.18 if np.isfinite(np.nanmax(hi.values)) else 5.0)
            for yi, v, h in zip(y, means.values, hi.values):
                if np.isfinite(v):
                    ax.text((h if np.isfinite(h) else v) + 0.015 * xmax, yi, f"{v:.1f}", va="center", ha="left",
                            fontsize=8.5, color=INK)
            ax.set_yticks(y, [label(x) for x in methods])
            ax.set_xlabel(METRIC_LABELS.get(m, m))
            ax.set_xlim(0, xmax)
            ax.grid(axis="y", visible=False)
        fig.suptitle(f"Odia QA quality on the real {split} split (95% passage-clustered bootstrap CI)",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout()
        written += _save(fig, out_dir, f"qa_scores_{split}")
    return written


# --------------------------------------------------------------------------------------
# 2. Predicted vs actual training time per run (dumbbell)
# --------------------------------------------------------------------------------------
def plot_training_time(train: pd.DataFrame, out_dir: Path) -> list[Path]:
    if train.empty or "actual_wall_clock_sec" not in train.columns:
        return []
    t = train.copy()
    t["_o"] = t["method"].map({m: i for i, m in enumerate(method_order(t["method"]))})
    t = t.sort_values(["_o", "run_id"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.5))
    y = np.arange(len(t))
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.plot([r["predicted_training_time_sec"], r["actual_wall_clock_sec"]], [yi, yi], color=GRID, lw=3, zorder=1)
        ax.scatter(r["actual_wall_clock_sec"], yi, s=70, color=color(r["method"]), zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.scatter(r["predicted_training_time_sec"], yi, s=70, facecolor=SURFACE, edgecolor=color(r["method"]),
                   linewidth=2, zorder=3)
        err = r.get("proxy_error_pct")
        if pd.notna(err):
            ax.text(max(r["predicted_training_time_sec"], r["actual_wall_clock_sec"]) * 1.03, yi,
                    f"{err:+.0f}%", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, t["run_id"])
    ax.set_xlabel("Training time (s)")
    ax.scatter([], [], s=70, color=INK_2, label="Actual (measured)")
    ax.scatter([], [], s=70, facecolor=SURFACE, edgecolor=INK_2, linewidth=2, label="Predicted by time proxy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    ax.set_title("Time proxy: predicted vs actual training time (label = signed proxy error)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, float(t[["predicted_training_time_sec", "actual_wall_clock_sec"]].max().max()) * 1.15)
    fig.tight_layout()
    return _save(fig, out_dir, "training_time_predicted_vs_actual")


# --------------------------------------------------------------------------------------
# 3. Micro-batch parity plot of the proxy
# --------------------------------------------------------------------------------------
def plot_proxy_step_parity(steps: pd.DataFrame, out_dir: Path, summary: dict | None = None) -> list[Path]:
    if steps.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for m in method_order(steps["method"]):
        s = steps[steps["method"] == m]
        ax.scatter(s["measured_ms"], s["proxy_predicted_ms"], s=14, alpha=0.6, color=color(m), label=label(m),
                   edgecolor="none")
    lim = [0, float(max(steps["measured_ms"].max(), steps["proxy_predicted_ms"].max()) * 1.05)]
    ax.plot(lim, lim, ls="--", lw=1.2, color=INK_2, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Measured micro-batch time (ms)")
    ax.set_ylabel("Proxy-predicted micro-batch time (ms)")
    title = "Proxy parity per micro-batch"
    if summary and "micro_batch_level" in summary:
        s = summary["micro_batch_level"]
        title += f"  (MAPE {s['mape_pct']:.1f}%, Spearman {s['spearman']:.2f})"
    ax.set_title(title, loc="left")

    ax = axes[1]
    for m in method_order(steps["method"]):
        s = steps[steps["method"] == m]
        ax.scatter(s["padded_seq_len"], s["measured_ms"] / s["proxy_predicted_ms"], s=14, alpha=0.6,
                   color=color(m), edgecolor="none", label=label(m))
    ax.axhline(1.0, ls="--", lw=1.2, color=INK_2)
    ax.set_xlabel("Padded sequence length of the micro-batch (tokens)")
    ax.set_ylabel("Measured / predicted")
    ax.set_title("Where the proxy is off: ratio vs sequence length", loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncols=min(4, len(labels)), markerscale=1.6)
    return _save(fig, out_dir, "proxy_step_parity")


# --------------------------------------------------------------------------------------
# 4. Training loss curves
# --------------------------------------------------------------------------------------
def plot_loss_curves(loss: pd.DataFrame, out_dir: Path, smooth: int = 5) -> list[Path]:
    if loss.empty:
        return []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_id, s in loss.groupby("run_id"):
        s = s.sort_values("step")
        y = s["loss"].rolling(smooth, min_periods=1).mean()
        m = config.method_of_run(run_id)
        ax.plot(s["step"], y, color=color(m), label=run_id)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel(f"Training loss (rolling mean, {smooth} steps)")
    ax.set_title("Training loss on the answer tokens", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "training_loss_curves")


# --------------------------------------------------------------------------------------
# 5-7. Selection diagnostics (no GPU needed)
# --------------------------------------------------------------------------------------
def plot_selection_overview(sel: pd.DataFrame, out_dir: Path, budget_sec: float | None = None) -> list[Path]:
    if sel.empty:
        return []
    s = sel if budget_sec is None else sel[np.isclose(sel["budget_sec"], budget_sec)]
    if s.empty:
        return []
    budget = float(s["budget_sec"].iloc[0])
    s = s[s["budget_sec"] == budget].groupby("method").mean(numeric_only=True)
    methods = method_order(s.index)
    panels = [("num_selected", "Examples selected"), ("total_utility", "Total utility  Σ u_i"),
              ("mean_token_length", "Mean sequence length (tokens)"), ("num_unique_passages", "Distinct wiki articles")]
    panels = [p for p in panels if p[0] in s.columns]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 0.5 * len(methods) + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(methods))[::-1]
    for ax, (col, title) in zip(axes, panels):
        vals = s[col].reindex(methods).values
        ax.barh(y, vals, height=0.62, color=[color(m) for m in methods], edgecolor=SURFACE, linewidth=2)
        for yi, v in zip(y, vals):
            if np.isfinite(v):
                ax.text(v, yi, f" {v:,.0f}" if v >= 10 else f" {v:.2f}", va="center", fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_yticks(y, [label(m) for m in methods])
        ax.set_xlim(0, np.nanmax(vals) * 1.25 if np.isfinite(np.nanmax(vals)) and np.nanmax(vals) > 0 else 1)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"What each method selects under the same {budget:.0f} s proxy training budget",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"selection_overview_b{config.budget_tag(budget)}")


def plot_token_length_distribution(subsets: dict[str, pd.DataFrame], out_dir: Path, budget_sec: float) -> list[Path]:
    subsets = {m: d for m, d in subsets.items() if len(d)}
    if not subsets:
        return []
    methods = method_order(subsets)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(methods) + 1.5))
    data = [subsets[m]["token_length"].values for m in methods]
    y = np.arange(len(methods))[::-1]
    bp = ax.boxplot(data, positions=y, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.5})
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(color(m))
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(2)
    ax.set_yticks(y, [f"{label(m)} (n={len(subsets[m])})" for m in methods])
    ax.set_xlabel("Training sequence length (tokens, prompt + answer)")
    ax.set_title(f"Length of selected examples (budget {budget_sec:.0f} s; whiskers 1.5 IQR)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, f"selection_token_lengths_b{config.budget_tag(budget_sec)}")


def plot_utility_vs_cost(pool: pd.DataFrame, subsets: dict[str, pd.DataFrame], out_dir: Path,
                         budget_sec: float, methods=("quality_only", "proposed_knapsack")) -> list[Path]:
    methods = [m for m in methods if m in subsets and len(subsets[m])]
    if pool.empty or not methods:
        return []
    cand = pool[pool["cm_ranked"]] if "cm_ranked" in pool.columns else pool
    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 4.4), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, methods):
        ax.scatter(cand["predicted_time_cost_ms"] / 1000, cand["utility"], s=5, color=POOL_GREY, alpha=0.5,
                   edgecolor="none", label=f"Candidates (n={len(cand):,})")
        sub = subsets[m]
        ax.scatter(sub["predicted_time_cost_ms"] / 1000, sub["utility"], s=9, color=color(m), alpha=0.85,
                   edgecolor="none", label=f"{label(m)} (n={len(sub):,})")
        ax.set_xscale("log")
        ax.set_xlabel("Proxy cost per example c_i (s, log scale)")
        ax.set_title(label(m), loc="left", fontsize=10)
        ax.legend(loc="lower right", markerscale=2.5)
    axes[0].set_ylabel("Composite utility u_i")
    fig.suptitle(f"Usefulness vs time cost (budget {budget_sec:.0f} s): the knapsack favours high u_i / c_i",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"utility_vs_cost_b{config.budget_tag(budget_sec)}")


def plot_budget_sweep(sweep: pd.DataFrame, out_dir: Path, y: str = "total_utility",
                      ylabel: str = "Total utility of the selected subset  Σ u_i") -> list[Path]:
    if sweep.empty or y not in sweep.columns:
        return []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xcol = "budget_x_real" if "budget_x_real" in sweep.columns else "budget_sec"
    g = sweep.groupby(["method", xcol])[y].mean().reset_index()
    for m in method_order(g["method"]):
        s = g[g["method"] == m].sort_values(xcol)
        ax.plot(s[xcol], s[y], marker="o", ms=6, color=color(m), label=label(m),
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    _log_budget_axis(ax, sorted(g[xcol].unique()), relative=xcol == "budget_x_real")
    ax.set_ylabel(ylabel)
    ax.set_title("Budget sweep (selection only, no training needed)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, f"budget_sweep_{y}")


def plot_covariance_shift(sel: pd.DataFrame, out_dir: Path) -> list[Path]:
    if sel.empty or "cov_shift_rel" not in sel.columns or sel["cov_shift_rel"].isna().all():
        return []
    return plot_budget_sweep(sel.dropna(subset=["cov_shift_rel"]), out_dir, y="cov_shift_rel",
                             ylabel="‖Cov(selected) − Cov(real)‖_F / ‖Cov(real)‖_F  (lower is better)")


# --------------------------------------------------------------------------------------
# 8-10. Joint quality / cost views
# --------------------------------------------------------------------------------------
def plot_quality_vs_time(table: pd.DataFrame, out_dir: Path, metric: str = "token_f1") -> list[Path]:
    if table.empty or "actual_wall_clock_sec" not in table.columns:
        return []
    written = []
    for split, t in table.groupby("split"):
        t = t.dropna(subset=["actual_wall_clock_sec", metric])
        if t.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for m in method_order(t["method"]):
            s = t[t["method"] == m]
            x = s["actual_wall_clock_sec"].to_numpy(dtype=float)
            v = s[metric].to_numpy(dtype=float)
            lo = s[f"{metric}_ci_low"].to_numpy(dtype=float)
            hi = s[f"{metric}_ci_high"].to_numpy(dtype=float)
            ax.errorbar(x, v, yerr=np.vstack([v - lo, hi - v]),
                        fmt="o", ms=8, color=color(m), ecolor=color(m), elinewidth=1.2, capsize=3, label=label(m),
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlabel("Actual training time (s)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Quality vs training cost ({split}); up-and-left is better", loc="left")
        ax.legend(loc="best")
        ax.set_xlim(left=0)
        fig.tight_layout()
        written += _save(fig, out_dir, f"quality_vs_training_time_{split}")
    return written


def plot_metric_vs_budget(table: pd.DataFrame, out_dir: Path, metric: str = "token_f1") -> list[Path]:
    if table.empty or "budget_sec" not in table.columns:
        return []
    written = []
    for split, t in table.dropna(subset=["budget_sec"]).groupby("split"):
        if t["budget_sec"].nunique() < 2:
            continue
        g = t.groupby(["method", "budget_sec"]).agg(v=(metric, "mean"), lo=(f"{metric}_ci_low", "mean"),
                                                    hi=(f"{metric}_ci_high", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for m in method_order(g["method"]):
            s = g[g["method"] == m].sort_values("budget_sec")
            ax.fill_between(s["budget_sec"], s["lo"], s["hi"], color=color(m), alpha=0.12, linewidth=0)
            ax.plot(s["budget_sec"], s["v"], marker="o", color=color(m), label=label(m))
        _log_budget_axis(ax, sorted(g["budget_sec"].unique()), relative=False)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs budget ({split}; band = 95% CI)", loc="left")
        ax.legend(loc="best")
        fig.tight_layout()
        written += _save(fig, out_dir, f"{metric}_vs_budget_{split}")
    return written


def plot_f1_by_answer_length(preds: dict[str, pd.DataFrame], out_dir: Path, split: str,
                             metric: str = "token_f1") -> list[Path]:
    preds = {k: v for k, v in preds.items() if v is not None and len(v)}
    if not preds:
        return []
    bins = [0, 1, 2, 4, 1000]
    names = ["1 word", "2 words", "3-4 words", "5+ words"]
    rows = []
    for run_id, p in preds.items():
        nwords = p["gold_answers"].map(lambda a: min(len(str(x).split()) for x in a) if len(a) else 0)
        bucket = pd.cut(nwords, bins=bins, labels=names, right=True)
        g = p.assign(bucket=bucket).groupby("bucket", observed=False)[metric].agg(["mean", "size"])
        for b, r in g.iterrows():
            rows.append({"run_id": run_id, "method": config.method_of_run(run_id), "bucket": b,
                         "value": 100 * r["mean"], "n": int(r["size"])})
    df = pd.DataFrame(rows)
    runs = sorted(preds, key=lambda r: method_order(df["method"]).index(config.method_of_run(r)))
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.8 / len(runs)
    x = np.arange(len(names))
    for i, r in enumerate(runs):
        s = df[df["run_id"] == r].set_index("bucket").reindex(names)
        ax.bar(x + (i - (len(runs) - 1) / 2) * width, s["value"].values, width=width, edgecolor=SURFACE,
               linewidth=2, color=color(config.method_of_run(r)), label=r)
    counts = df.groupby("bucket", observed=False)["n"].max().reindex(names)
    ax.set_xticks(x, [f"{n}\n(n={int(c) if pd.notna(c) else 0})" for n, c in zip(names, counts)])
    ax.set_xlabel("Length of the gold answer")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by answer length ({split})", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out_dir, f"{metric}_by_answer_length_{split}")


def plot_pipeline_overhead(components: pd.DataFrame, out_dir: Path) -> list[Path]:
    """components: rows (method, component, seconds). Stacked horizontal bars.
    Components are distinguished by lightness steps of one neutral ramp plus direct labels,
    since the method colours already carry identity elsewhere."""
    if components.empty:
        return []
    comps = list(dict.fromkeys(components["component"]))
    ramp = ["#0d366b", "#1c5cab", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb", "#e4e3df"][:len(comps)]
    methods = method_order(components["method"])
    pivot = components.pivot_table(index="method", columns="component", values="seconds", aggfunc="sum") \
        .reindex(index=methods, columns=comps).fillna(0.0)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(methods) + 1.8))
    y = np.arange(len(methods))[::-1]
    left = np.zeros(len(methods))
    for comp, col in zip(comps, ramp):
        vals = pivot[comp].values
        ax.barh(y, vals, left=left, height=0.6, color=col, edgecolor=SURFACE, linewidth=2, label=comp)
        left += vals
    for yi, tot in zip(y, left):
        ax.text(tot, yi, f"  {tot:,.0f} s", va="center", fontsize=8.5)
    ax.set_yticks(y, [label(m) for m in methods])
    ax.set_xlabel("Wall-clock seconds")
    ax.set_title("End-to-end cost: selection pipeline overhead + training", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=min(4, len(comps)))
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, left.max() * 1.15 if left.max() > 0 else 1)
    fig.tight_layout()
    return _save(fig, out_dir, "pipeline_overhead")


def plot_proxy_profile_fit(profile_csv: Path, proxy, out_dir: Path) -> list[Path]:
    """Re-plot of the profiling grid with the fitted proxy curves (for the report)."""
    if not Path(profile_csv).is_file():
        return []
    df = pd.read_csv(profile_csv)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for col, (b, s) in zip(palette, df.groupby("batch_size")):
        s = s.sort_values("seq_len")
        ax.scatter(s["seq_len"], s["step_time_ms"] / 1000, s=40, color=col, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=f"B={b} measured")
        grid = np.linspace(s["seq_len"].min(), s["seq_len"].max(), 100)
        ax.plot(grid, proxy.step_ms(b, grid) / 1000, color=col, lw=1.5, ls="--")
    ax.set_xlabel("Sequence length L (tokens)")
    ax.set_ylabel("Step time (s)")
    ax.set_title(f"Time proxy fit on the profiling grid ({proxy.gpu_name}); dashed = proxy", loc="left")
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return _save(fig, out_dir, "proxy_profile_fit")


def plot_covariance_diagnostic(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One figure per budget: relative covariance distance of each method's subset to the real
    data (PCA space and full space), with the selection stages' starting sets as grey references."""
    if table.empty:
        return []
    subsets = table[table["kind"] == "subset"]
    refs = table[table["kind"] == "reference"]
    written = []
    panels = [("rel_dist_pca", "PCA space (what covariance matching optimises)"),
              ("rel_dist_full", "Full embedding space")]
    for budget, sb in subsets.groupby("budget_sec", dropna=False):
        g = sb.groupby("method")[[c for c, _ in panels]].mean()
        methods = method_order(g.index)
        labels = [label(m) for m in methods] + list(refs["run_id"])
        colors = [color(m) for m in methods] + [POOL_GREY] * len(refs)
        fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(labels) + 1.8), sharey=True)
        y = np.arange(len(labels))[::-1]
        for ax, (col, title) in zip(axes, panels):
            vals = np.r_[g[col].reindex(methods).values, refs[col].values]
            ax.barh(y, vals, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
            xmax = float(np.nanmax(vals)) * 1.2 if np.isfinite(np.nanmax(vals)) else 1.0
            for yi, v in zip(y, vals):
                if np.isfinite(v):
                    ax.text(v + 0.01 * xmax, yi, f"{v:.3f}", va="center", fontsize=8.5)
            if len(refs):
                ax.axhline(y[len(methods) - 1] - 0.5, color=INK_2, lw=0.8, ls=":")
            ax.set_xlim(0, xmax)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Relative covariance distance (lower = closer to real data)")
            ax.set_title(title, loc="left", fontsize=10)
            ax.grid(axis="y", visible=False)
        btxt = f"budget {budget:.0f} s" if np.isfinite(budget) else "all budgets"
        fig.suptitle(f"Covariance shift of each selected subset vs the real train data ({btxt})\n"
                     "distance = ‖Cov(subset) − Cov(real)‖_F / ‖Cov(real)‖_F;  grey = reference sets",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout()
        tag = config.budget_tag(budget) if np.isfinite(budget) else "all"
        written += _save(fig, out_dir, f"covariance_shift_b{tag}")
    return written
