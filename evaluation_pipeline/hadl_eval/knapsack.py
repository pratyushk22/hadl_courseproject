"""0/1 knapsack solvers for budgeted data selection.

maximise   sum_i u_i x_i
subject to sum_i c_i x_i <= B,   x_i in {0, 1}

* greedy_density : sort by u_i / c_i (classic 1/2-approximation when combined with the
                   best single item); O(n log n).
* dp             : exact dynamic programme on costs discretised to `resolution` units.
                   Costs are rounded UP, so every DP solution is feasible for the real costs.
* solve          : runs both and returns the better feasible solution, then fills any
                   leftover budget greedily.
"""
from __future__ import annotations

import numpy as np


def _validate(utility, cost, budget):
    u = np.asarray(utility, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    if u.shape != c.shape or u.ndim != 1:
        raise ValueError("utility and cost must be 1-D arrays of the same length")
    if not (np.isfinite(u).all() and np.isfinite(c).all()):
        raise ValueError("utility and cost must be finite")
    if (c <= 0).any():
        raise ValueError("costs must be strictly positive")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    return u, c


def fill_remaining(selected: np.ndarray, utility, cost, budget: float) -> np.ndarray:
    """Add unselected items with positive utility (best density first) while they fit."""
    u, c = _validate(utility, cost, budget)
    chosen = np.zeros(len(u), dtype=bool)
    chosen[np.asarray(selected, dtype=np.int64)] = True
    spent = c[chosen].sum()
    for i in np.argsort(-(u / c), kind="mergesort"):
        if not chosen[i] and u[i] > 0 and spent + c[i] <= budget:
            chosen[i] = True
            spent += c[i]
    return np.flatnonzero(chosen)


def greedy_density(utility, cost, budget: float) -> np.ndarray:
    u, c = _validate(utility, cost, budget)
    order = np.argsort(-(u / c), kind="mergesort")
    chosen, spent = [], 0.0
    for i in order:
        if u[i] <= 0:
            break
        if spent + c[i] <= budget:
            chosen.append(i)
            spent += c[i]
    chosen = np.array(chosen, dtype=np.int64)
    # 1/2-approximation guard: compare with the best single feasible item.
    feasible = np.flatnonzero((c <= budget) & (u > 0))
    if feasible.size:
        best_single = feasible[np.argmax(u[feasible])]
        if u[best_single] > u[chosen].sum():
            chosen = np.array([best_single], dtype=np.int64)
    return np.sort(chosen)


def dp(utility, cost, budget: float, resolution: int = 2000) -> np.ndarray:
    """Exact DP over discretised costs. Memory O(n * resolution) bits."""
    u, c = _validate(utility, cost, budget)
    n = len(u)
    if n == 0 or budget <= 0:
        return np.array([], dtype=np.int64)
    unit = budget / resolution
    w = np.ceil(c / unit - 1e-9).astype(np.int64)
    w = np.maximum(w, 1)
    W = resolution
    best = np.zeros(W + 1)
    take = np.zeros((n, W + 1), dtype=bool)
    for i in range(n):
        wi = w[i]
        if wi > W or u[i] <= 0:
            continue
        candidate = best[:W + 1 - wi] + u[i]
        improve = candidate > best[wi:]
        take[i, wi:] = improve
        best[wi:] = np.where(improve, candidate, best[wi:])
    # backtrack
    cap = int(np.argmax(best))
    chosen = []
    for i in range(n - 1, -1, -1):
        if cap >= 0 and take[i, cap]:
            chosen.append(i)
            cap -= w[i]
    chosen = np.array(sorted(chosen), dtype=np.int64)
    assert c[chosen].sum() <= budget + 1e-6, "DP produced an infeasible solution"
    return chosen


def solve(utility, cost, budget: float, method: str = "auto", resolution: int = 2000,
          max_dp_cells: int = 60_000_000) -> tuple[np.ndarray, dict]:
    """Return (selected indices, info). method in {'auto', 'dp', 'greedy'}."""
    u, c = _validate(utility, cost, budget)
    info = {"solver": method, "resolution": resolution}
    candidates = {"greedy": greedy_density(u, c, budget)}
    use_dp = method == "dp" or (method == "auto" and len(u) * (resolution + 1) <= max_dp_cells)
    if use_dp:
        candidates["dp"] = dp(u, c, budget, resolution=resolution)
    for k in list(candidates):
        candidates[k] = fill_remaining(candidates[k], u, c, budget)
    values = {k: float(u[v].sum()) for k, v in candidates.items()}
    if method in ("dp", "greedy"):
        winner = method
    else:
        winner = max(values, key=values.get)
    info.update({"objective_" + k: v for k, v in values.items()})
    info["winner"] = winner
    return candidates[winner], info
