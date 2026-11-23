"""Pipeline integration: build one candidate table from the teammates' outputs.

For every unique synthetic row of the low-rank pool we attach
    lr_score      low-rank usefulness. The low-rank stage sampled rows with
                  p_i = score_i / sum(score), score_i = max(pred_loss, 0) + lambda*residual_sq,
                  so `sampling_probability` is exactly proportional to its usefulness score.
    lr_pct        percentile rank of lr_score in (0, 1]
    cm_rank       position in the greedy covariance-matching order (NaN if not ranked)
    cm_score      1 - cm_rank / N_ranked in (0, 1]; 0 for unranked rows
    utility       alpha * lr_pct + (1 - alpha) * cm_score   (composite usefulness u_i)
    token_length, predicted_time_cost_ms   from the time proxy (cost c_i)
    overlaps_eval contamination flag (shares an 8-word shingle with a real val/test passage)

The random baseline needs costs too; it draws from the *whole* synthetic training pool
(train_85_percent.parquet, 185k rows) - a uniformly shuffled slice of RANDOM_POOL_SIZE rows
is costed, which is statistically identical to shuffling the full pool for any budget that
the slice can fill.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import load_real_split, standardize_synthetic
from .text_utils import word_shingles


def find_cm_ranking(cm_dir: str | Path = config.CM_OUTPUT_DIR) -> Path:
    """Largest covariance_matched_n*.parquet (greedy is nested, so it contains all smaller ones)."""
    files = sorted(Path(cm_dir).glob("covariance_matched_n*.parquet"),
                   key=lambda p: int(p.stem.split("_n")[-1]))
    if not files:
        raise FileNotFoundError(f"No covariance_matched_n*.parquet in {cm_dir}; run covariance_matching.py first")
    return files[-1]


def percentile_rank(x: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1]; robust to the heavy tail of predicted losses."""
    s = pd.Series(np.asarray(x, dtype=np.float64))
    return (s.rank(method="average") / len(s)).to_numpy()


def eval_shingles(splits=("validation", "test"), n: int = 8) -> set[str]:
    shingles: set[str] = set()
    for split in splits:
        for passage in load_real_split(split)["passage"].unique():
            shingles |= word_shingles(passage, n)
    return shingles


def contamination_flags(passages, shingles: set[str], n: int = 8) -> np.ndarray:
    return np.array([bool(word_shingles(p, n) & shingles) for p in passages], dtype=bool)


def build_candidate_pool(alpha: float = config.DEFAULT_ALPHA, lr_draws_path=config.LR_DRAWS,
                         cm_path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Unique low-rank rows with LR + CM scores and composite utility (no costs yet)."""
    t0 = time.perf_counter()
    draws = pd.read_parquet(lr_draws_path)
    required = {"_source_row_idx", "sampling_probability", "predicted_loss", "residual_sq"}
    missing = required - set(draws.columns)
    if missing:
        raise KeyError(f"{lr_draws_path} lacks columns {missing}")
    draw_count = draws["_source_row_idx"].value_counts()
    pool = draws.drop_duplicates("_source_row_idx", keep="first").copy()
    pool["draw_count"] = pool["_source_row_idx"].map(draw_count).astype(int)
    pool["lr_score"] = pool["sampling_probability"].astype(np.float64)
    pool["lr_pct"] = percentile_rank(pool["lr_score"].values)

    cm_path = Path(cm_path) if cm_path else find_cm_ranking()
    cm = pd.read_parquet(cm_path, columns=["_source_row_idx", "cm_rank"])
    if cm["_source_row_idx"].duplicated().any():
        raise ValueError(f"{cm_path} has duplicated _source_row_idx")
    n_ranked = len(cm)
    pool = pool.merge(cm, on="_source_row_idx", how="left")
    unmatched = n_ranked - int(pool["cm_rank"].notna().sum())
    if unmatched:
        raise ValueError(f"{unmatched} covariance-matched rows are not in the low-rank pool - inputs out of sync")
    pool["cm_ranked"] = pool["cm_rank"].notna()
    pool["cm_score"] = np.where(pool["cm_ranked"], 1.0 - pool["cm_rank"].fillna(0).values / n_ranked, 0.0)
    pool["utility"] = alpha * pool["lr_pct"] + (1.0 - alpha) * pool["cm_score"]
    pool = standardize_synthetic(pool).reset_index(drop=True)

    info = {
        "lr_draws_file": str(lr_draws_path),
        "cm_ranking_file": str(cm_path),
        "n_draws": int(len(draws)),
        "n_unique_candidates": int(len(pool)),
        "n_cm_ranked": int(n_ranked),
        "alpha": alpha,
        "build_seconds": time.perf_counter() - t0,
    }
    return pool, info


def build_random_pool(size: int = config.RANDOM_POOL_SIZE, seed: int = 0,
                      source_path=config.LR_TRAIN_POOL) -> pd.DataFrame:
    """Uniformly shuffled slice of the full synthetic training pool (for the random baseline)."""
    full = pd.read_parquet(source_path)
    rng = np.random.default_rng(seed)
    take = rng.permutation(len(full))[:min(size, len(full))]
    out = standardize_synthetic(full.iloc[take].copy())
    out["random_pool_order"] = np.arange(len(out))
    return out.reset_index(drop=True)
