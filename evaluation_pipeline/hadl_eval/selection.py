"""Budgeted subset selection for every method compared in the project.

All methods receive the same synthetic training-time budget B (proxy milliseconds per
epoch) and return a subset whose predicted cost is <= B.

    random             uniform random order over the full synthetic train pool; prefix fill
    low_rank_only      ablation: highest low-rank score first (whole LR pool); prefix fill
    cov_match_only     ablation: greedy covariance-matching order; prefix fill
    quality_only       baseline "two methods combined": composite utility u_i (LR + CM),
                       highest first, ignoring cost; prefix fill
    proposed_knapsack  our method: max sum u_i s.t. sum c_i <= B (0/1 knapsack, exact DP +
                       greedy density, best of both), using the time proxy cost c_i

"Prefix fill" stops at the first example that would exceed the budget, i.e. the method
really ignores cost. Only the proposed method trades usefulness against cost.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import knapsack

COST_COL = "predicted_time_cost_ms"


def prefix_within_budget(order: np.ndarray, cost: np.ndarray, budget_ms: float) -> np.ndarray:
    """Longest prefix of `order` whose cumulative cost is <= budget."""
    order = np.asarray(order, dtype=np.int64)
    cum = np.cumsum(np.asarray(cost, dtype=np.float64)[order])
    n = int(np.searchsorted(cum, budget_ms, side="right"))
    return order[:n]


def _finish(df: pd.DataFrame, idx: np.ndarray, method: str, seconds: float, extra: dict | None = None):
    sub = df.iloc[np.asarray(idx, dtype=np.int64)].copy()
    sub["selection_method"] = method
    sub["selection_order"] = np.arange(len(sub))
    info = {"method": method, "selection_seconds": seconds}
    info.update(extra or {})
    return sub.reset_index(drop=True), info


def select_random(random_pool: pd.DataFrame, budget_ms: float, seed: int):
    t0 = time.perf_counter()
    order = np.random.default_rng(seed).permutation(len(random_pool))
    idx = prefix_within_budget(order, random_pool[COST_COL].values, budget_ms)
    if len(idx) == len(random_pool) and random_pool[COST_COL].sum() > 0:
        print("[select] WARNING: random pool exhausted before the budget was filled; "
              "increase --random_pool_size")
    return _finish(random_pool, idx, "random", time.perf_counter() - t0)


def select_low_rank_only(pool: pd.DataFrame, budget_ms: float, seed: int = 0):
    t0 = time.perf_counter()
    order = np.argsort(-pool["lr_score"].values, kind="mergesort")
    idx = prefix_within_budget(order, pool[COST_COL].values, budget_ms)
    return _finish(pool, idx, "low_rank_only", time.perf_counter() - t0)


def select_cov_match_only(pool: pd.DataFrame, budget_ms: float, seed: int = 0):
    t0 = time.perf_counter()
    ranked = np.flatnonzero(pool["cm_ranked"].values)
    order = ranked[np.argsort(pool["cm_rank"].values[ranked], kind="mergesort")]
    idx = prefix_within_budget(order, pool[COST_COL].values, budget_ms)
    if len(idx) == len(order):
        print("[select] WARNING: covariance-matching ranking exhausted before the budget was filled; "
              "re-run covariance_matching.py with larger --multiples")
    return _finish(pool, idx, "cov_match_only", time.perf_counter() - t0)


def _candidates(pool: pd.DataFrame, candidate_set: str) -> np.ndarray:
    if candidate_set == "cm_ranked":
        return np.flatnonzero(pool["cm_ranked"].values)
    if candidate_set == "lr_pool":
        return np.arange(len(pool))
    raise ValueError("candidate_set must be 'cm_ranked' or 'lr_pool'")


def select_quality_only(pool: pd.DataFrame, budget_ms: float, seed: int = 0, candidate_set: str = "cm_ranked"):
    t0 = time.perf_counter()
    cand = _candidates(pool, candidate_set)
    order = cand[np.argsort(-pool["utility"].values[cand], kind="mergesort")]
    idx = prefix_within_budget(order, pool[COST_COL].values, budget_ms)
    exhausted = len(idx) == len(cand)
    if exhausted:
        print(f"[select] WARNING quality_only: all {len(cand)} candidates fit the budget - the ranking is "
              "exhausted, so the budget is not binding (re-run covariance matching with larger --multiples)")
    return _finish(pool, idx, "quality_only", time.perf_counter() - t0,
                   {"candidate_set": candidate_set, "candidates_exhausted": exhausted})


def select_proposed_knapsack(pool: pd.DataFrame, budget_ms: float, seed: int = 0,
                             candidate_set: str = "cm_ranked", solver: str = "auto", resolution: int = 2000):
    t0 = time.perf_counter()
    cand = _candidates(pool, candidate_set)
    u = pool["utility"].values[cand]
    c = pool[COST_COL].values[cand]
    chosen, info = knapsack.solve(u, c, budget_ms, method=solver, resolution=resolution)
    idx = cand[chosen]
    # Present the subset in decreasing utility order (training shuffles anyway).
    idx = idx[np.argsort(-pool["utility"].values[idx], kind="mergesort")]
    info["candidate_set"] = candidate_set
    info["candidates_exhausted"] = len(idx) == len(cand)
    if info["candidates_exhausted"]:
        print(f"[select] WARNING proposed_knapsack: all {len(cand)} candidates fit the budget - the budget is "
              "not binding (re-run covariance matching with larger --multiples or use --candidate_set lr_pool)")
    return _finish(pool, idx, "proposed_knapsack", time.perf_counter() - t0, info)


SELECTORS = {
    "random": select_random,
    "low_rank_only": select_low_rank_only,
    "cov_match_only": select_cov_match_only,
    "quality_only": select_quality_only,
    "proposed_knapsack": select_proposed_knapsack,
}


def run_selection(method: str, pool: pd.DataFrame, random_pool: pd.DataFrame, budget_ms: float,
                  seed: int, **kwargs):
    if method not in SELECTORS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(SELECTORS)}")
    if method == "random":
        return select_random(random_pool, budget_ms, seed)
    if method in ("quality_only", "proposed_knapsack"):
        return SELECTORS[method](pool, budget_ms, seed, **kwargs)
    return SELECTORS[method](pool, budget_ms, seed)


def subset_stats(sub: pd.DataFrame, budget_ms: float) -> dict:
    n = len(sub)
    cost = sub[COST_COL].values if n else np.array([0.0])
    stats = {
        "num_selected": int(n),
        "budget_sec": budget_ms / 1000.0,
        "predicted_epoch_time_sec": float(cost.sum() / 1000.0) if n else 0.0,
        "budget_used_pct": float(cost.sum() / budget_ms * 100.0) if (n and budget_ms > 0) else 0.0,
        "mean_token_length": float(sub["token_length"].mean()) if n else 0.0,
        "median_token_length": float(sub["token_length"].median()) if n else 0.0,
        "total_tokens": int(sub["token_length"].sum()) if n else 0,
        "num_unique_passages": int(sub["passage_id"].nunique()) if n else 0,
    }
    for col in ("utility", "lr_pct", "cm_score"):
        if col in sub.columns and n:
            stats[f"total_{col}"] = float(sub[col].sum())
            stats[f"mean_{col}"] = float(sub[col].mean())
    if "overlaps_eval" in sub.columns and n:
        stats["num_overlapping_eval_passages"] = int(sub["overlaps_eval"].sum())
    return stats
