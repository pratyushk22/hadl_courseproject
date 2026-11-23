"""Evaluation metrics.

QA quality (per example, max over gold answers):
    * exact_match      - normalised exact match (EM)
    * token_f1         - SQuAD token-overlap F1 on whitespace tokens
    * char_f1          - character-overlap F1; robust to Odia agglutinative suffixes
                         (e.g. gold "ଭୁବନେଶ୍ୱର" vs prediction "ଭୁବନେଶ୍ୱରରେ")
    * contains_gold    - normalised gold answer appears inside the prediction

Statistics:
    * bootstrap confidence intervals (optionally clustered by passage)
    * paired (clustered) bootstrap significance test between two systems

Training-time proxy:
    * absolute / relative error, MAPE, RMSE, R^2, Spearman rank correlation
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .text_utils import normalize_answer

QA_METRICS = ["exact_match", "token_f1", "char_f1", "contains_gold"]


# --------------------------------------------------------------------------------------
# Per-example QA metrics
# --------------------------------------------------------------------------------------
def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def _overlap_f1(pred_items: Sequence, gold_items: Sequence) -> float:
    if not pred_items and not gold_items:
        return 1.0
    if not pred_items or not gold_items:
        return 0.0
    common = Counter(pred_items) & Counter(gold_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_items)
    recall = num_same / len(gold_items)
    return 2.0 * precision * recall / (precision + recall)


def token_f1(pred: str, gold: str) -> float:
    return _overlap_f1(normalize_answer(pred).split(), normalize_answer(gold).split())


def char_f1(pred: str, gold: str) -> float:
    return _overlap_f1(list(normalize_answer(pred).replace(" ", "")),
                       list(normalize_answer(gold).replace(" ", "")))


def contains_gold(pred: str, gold: str) -> float:
    g = normalize_answer(gold)
    return float(bool(g) and g in normalize_answer(pred))


def score_example(pred: str, golds: Iterable[str]) -> dict:
    """All QA metrics for one example; each metric is the max over the gold answers."""
    golds = [g for g in (golds or []) if g is not None and str(g).strip()] or [""]
    return {
        "exact_match": max(exact_match(pred, g) for g in golds),
        "token_f1": max(token_f1(pred, g) for g in golds),
        "char_f1": max(char_f1(pred, g) for g in golds),
        "contains_gold": max(contains_gold(pred, g) for g in golds),
    }


# --------------------------------------------------------------------------------------
# Bootstrap statistics
# --------------------------------------------------------------------------------------
def _cluster_index(groups: Sequence | None, n: int):
    """Return (list of index arrays per cluster). Without groups every example is a cluster."""
    if groups is None:
        return [np.array([i]) for i in range(n)]
    groups = np.asarray(groups)
    if len(groups) != n:
        raise ValueError("groups must have the same length as values")
    _, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def _bootstrap_means(values_matrix: np.ndarray, clusters, n_boot: int, seed: int) -> np.ndarray:
    """values_matrix: (n_examples, k). Returns (n_boot, k) bootstrap means (cluster resampling,
    ratio estimator sum/count so clusters of different size are handled correctly)."""
    rng = np.random.default_rng(seed)
    n_clusters = len(clusters)
    sums = np.stack([values_matrix[c].sum(axis=0) for c in clusters])        # (C, k)
    counts = np.array([len(c) for c in clusters], dtype=np.float64)          # (C,)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    weights = np.zeros((n_boot, n_clusters))
    for b in range(n_boot):
        weights[b] = np.bincount(draws[b], minlength=n_clusters)
    return (weights @ sums) / (weights @ counts)[:, None]


def bootstrap_ci(values: Sequence[float], groups: Sequence | None = None, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Mean and (1-alpha) percentile bootstrap CI. Returns (mean, low, high)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    clusters = _cluster_index(groups, len(v))
    boots = _bootstrap_means(v[:, None], clusters, n_boot, seed)[:, 0]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap_test(a: Sequence[float], b: Sequence[float], groups: Sequence | None = None,
                          n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired bootstrap of mean(a) - mean(b) on the same examples.

    Returns the observed difference, its CI, and a two-sided p-value
    p = 2 * min(P*(diff <= 0), P*(diff >= 0)) (clipped to 1).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired test needs equally sized score vectors (same examples)")
    if a.size == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    diff = a - b
    clusters = _cluster_index(groups, len(diff))
    boots = _bootstrap_means(diff[:, None], clusters, n_boot, seed)[:, 0]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    p = 2.0 * min(np.mean(boots <= 0.0), np.mean(boots >= 0.0))
    return {"diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(min(1.0, p))}


def summarize_qa(per_example: list[dict], groups: Sequence | None = None, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Aggregate per-example metric dicts into percentages with 95% CIs."""
    out = {"num_examples": len(per_example)}
    for m in QA_METRICS:
        vals = [float(r[m]) for r in per_example]
        mean, lo, hi = bootstrap_ci(vals, groups=groups, n_boot=n_boot, seed=seed)
        out[m] = 100.0 * mean
        out[f"{m}_ci_low"] = 100.0 * lo
        out[f"{m}_ci_high"] = 100.0 * hi
    return out


# --------------------------------------------------------------------------------------
# Time-proxy error metrics
# --------------------------------------------------------------------------------------
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def proxy_error(predicted: Sequence[float], actual: Sequence[float]) -> dict:
    """Error of a time proxy. Relative errors are w.r.t. the actual (measured) time."""
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if p.shape != a.shape:
        raise ValueError("predicted and actual must have the same shape")
    mask = np.isfinite(p) & np.isfinite(a) & (a > 0)
    p, a = p[mask], a[mask]
    if p.size == 0:
        return {"n": 0, "mape_pct": float("nan"), "median_ape_pct": float("nan"), "rmse": float("nan"),
                "mean_signed_error_pct": float("nan"), "r2": float("nan"), "spearman": float("nan"),
                "scale_actual_over_pred": float("nan")}
    ape = np.abs(p - a) / a * 100.0
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return {
        "n": int(p.size),
        "mape_pct": float(ape.mean()),
        "median_ape_pct": float(np.median(ape)),
        "rmse": float(np.sqrt(np.mean((p - a) ** 2))),
        "mean_signed_error_pct": float(np.mean((p - a) / a) * 100.0),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": _spearman(p, a),
        # A proxy profiled on different hardware can be off by a constant factor but still
        # rank examples correctly; this factor is what a one-shot calibration would apply.
        "scale_actual_over_pred": float(np.sum(a) / np.sum(p)) if np.sum(p) > 0 else float("nan"),
    }


def loo_calibrated_error(predicted: Sequence[float], actual: Sequence[float]) -> dict:
    """Leave-one-out calibrated proxy error: for each run, fit a single scale factor on all
    other runs and measure the error of the rescaled prediction. Shows how good the proxy is
    once its (hardware-dependent) constant factor is known."""
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    n = len(p)
    if n < 2:
        return {"n": n, "loo_calibrated_mape_pct": float("nan")}
    ape = []
    for i in range(n):
        rest = np.arange(n) != i
        scale = a[rest].sum() / p[rest].sum() if p[rest].sum() > 0 else np.nan
        ape.append(abs(p[i] * scale - a[i]) / a[i] * 100.0)
    return {"n": n, "loo_calibrated_mape_pct": float(np.nanmean(ape))}
