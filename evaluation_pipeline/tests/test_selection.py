import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadl_eval import knapsack  # noqa: E402
from hadl_eval.selection import prefix_within_budget, run_selection, subset_stats  # noqa: E402
from hadl_eval.time_proxy import TimeProxy  # noqa: E402


def brute_force(u, c, B):
    best, best_set = 0.0, ()
    for r in range(len(u) + 1):
        for s in itertools.combinations(range(len(u)), r):
            if c[list(s)].sum() <= B and u[list(s)].sum() > best:
                best, best_set = u[list(s)].sum(), s
    return best


@pytest.mark.parametrize("seed", range(25))
def test_knapsack_near_optimal_and_feasible(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(3, 12)
    u = rng.random(n)
    c = rng.uniform(0.1, 1.0, n)
    B = float(c.sum() * rng.uniform(0.2, 0.7))
    sel, info = knapsack.solve(u, c, B, resolution=4000)
    assert c[sel].sum() <= B + 1e-9
    opt = brute_force(u, c, B)
    assert u[sel].sum() >= 0.97 * opt          # DP on rounded-up costs is (near) exact
    assert len(set(sel.tolist())) == len(sel)


def test_knapsack_edge_cases():
    assert len(knapsack.solve(np.array([1.0]), np.array([2.0]), 1.0)[0]) == 0
    sel, _ = knapsack.solve(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 10.0)
    assert sorted(sel.tolist()) == [0, 1]
    with pytest.raises(ValueError):
        knapsack.solve(np.array([1.0]), np.array([0.0]), 1.0)
    # greedy 1/2-approx guard: one big valuable item beats many small ones
    u = np.array([10.0, 1.0, 1.0])
    c = np.array([10.0, 1.0, 1.0])
    assert knapsack.greedy_density(u, c, 10.0).tolist() == [0]


def test_prefix_within_budget():
    order = np.array([2, 0, 1])
    cost = np.array([1.0, 5.0, 2.0])
    assert prefix_within_budget(order, cost, 3.0).tolist() == [2, 0]
    assert prefix_within_budget(order, cost, 2.9).tolist() == [2]   # strict prefix, no skipping
    assert prefix_within_budget(order, cost, 0.0).tolist() == []


def _toy_pool(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ranked = np.zeros(n, bool)
    ranked[:120] = True
    cm_rank = np.where(ranked, rng.permutation(n)[:n] % 120, np.nan)
    cm_rank[ranked] = rng.permutation(120)
    df = pd.DataFrame({
        "lr_score": rng.random(n), "lr_pct": rng.random(n), "cm_ranked": ranked, "cm_rank": cm_rank,
        "cm_score": np.where(ranked, 1 - np.nan_to_num(cm_rank) / 120, 0.0),
        "token_length": rng.integers(50, 2000, n), "passage_id": rng.integers(0, 40, n).astype(str),
        "_source_row_idx": np.arange(n),
    })
    df["utility"] = 0.5 * df["lr_pct"] + 0.5 * df["cm_score"]
    df["predicted_time_cost_ms"] = TimeProxy(0, 106.0, 3.08, 0.00156).per_sample_ms(df["token_length"].values)
    return df


@pytest.mark.parametrize("method", ["random", "quality_only", "proposed_knapsack", "low_rank_only", "cov_match_only"])
def test_every_method_respects_budget(method):
    pool = _toy_pool()
    budget = 0.15 * pool["predicted_time_cost_ms"].sum()
    sub, info = run_selection(method, pool, pool, budget, seed=3)
    assert sub["predicted_time_cost_ms"].sum() <= budget + 1e-6
    assert sub["_source_row_idx"].is_unique
    st = subset_stats(sub, budget)
    assert st["budget_used_pct"] <= 100.0 + 1e-9
    if method in ("quality_only", "proposed_knapsack", "cov_match_only"):
        assert sub["cm_ranked"].all()


def test_proposed_beats_quality_only_on_objective():
    pool = _toy_pool(seed=5)
    budget = 0.1 * pool["predicted_time_cost_ms"].sum()
    q, _ = run_selection("quality_only", pool, pool, budget, 0)
    p, _ = run_selection("proposed_knapsack", pool, pool, budget, 0)
    assert p["utility"].sum() >= q["utility"].sum()


def test_random_depends_on_seed_only():
    pool = _toy_pool()
    b = 0.1 * pool["predicted_time_cost_ms"].sum()
    a1, _ = run_selection("random", pool, pool, b, 1)
    a2, _ = run_selection("random", pool, pool, b, 1)
    a3, _ = run_selection("random", pool, pool, b, 2)
    assert a1["_source_row_idx"].tolist() == a2["_source_row_idx"].tolist()
    assert a1["_source_row_idx"].tolist() != a3["_source_row_idx"].tolist()


def test_time_proxy():
    p = TimeProxy(10.0, 1.0, 0.5, 0.001, max_seq_len=100)
    assert p.step_ms(2, 10) == pytest.approx(10 + 2 + 10 + 0.2)
    assert p.step_ms(1, 1000) == p.step_ms(1, 100)            # capped at max_seq_len
    assert p.per_sample_ms(10, batch_size=2) == pytest.approx(p.step_ms(2, 10) / 2)
    assert p.padded_batches_ms([10, 3, 5], batch_size=2, pad_to_multiple_of=8) == \
        pytest.approx(p.step_ms(2, 16) + p.step_ms(1, 8))
