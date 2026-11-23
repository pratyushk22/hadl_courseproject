"""Covariance-shift diagnostic for selected subsets.

For a set of synthetic rows S it reports

    rel_dist_pca  = ||Cov_k(S) - Cov_k(real)||_F / ||Cov_k(real)||_F   in the k-dim PCA space of the
                    real embeddings (k = 64, the space covariance matching optimises; same
                    conventions as covariance_matching.py: PCA fitted on the real rows, 1/n covariances)
    rel_dist_full = the same in the full embedding space (768-dim), which also covers the
                    directions covariance matching does not control

Lower is better (closer to the real-data covariance). Needs the git-ignored embedding files:
    embedding_generation/odia_embeddings.pt             all 216,394 synthetic rows (row = _source_row_idx)
    embedding_generation/odia_real_train_embeddings.pt  the 886 real IndicQA train rows
`try_create` returns None when they are missing, so the main pipeline still runs without them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config


def _load_matrix(path) -> np.ndarray:
    import torch

    t = torch.load(str(path), map_location="cpu", weights_only=True)
    if hasattr(t, "numpy"):
        t = t.float().numpy()
    arr = np.asarray(t)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D embedding matrix, got shape {arr.shape}")
    return arr


def _cov(x: np.ndarray) -> np.ndarray:
    """1/n covariance in float64 (matches covariance_matching.py)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean(axis=0)
    return xc.T @ xc / len(x)


class CovarianceDiagnostic:
    def __init__(self, synth_emb_path=config.SYNTH_EMBEDDINGS_PT, real_emb_path=config.REAL_TRAIN_EMBEDDINGS_PT,
                 pca_dim: int = 64, synth: np.ndarray | None = None, real: np.ndarray | None = None):
        self.synth = synth if synth is not None else _load_matrix(synth_emb_path)
        real = np.asarray(real if real is not None else _load_matrix(real_emb_path), dtype=np.float64)
        if real.shape[1] != self.synth.shape[1]:
            raise ValueError(f"embedding dims differ: real {real.shape[1]} vs synthetic {self.synth.shape[1]}")
        if not 0 < pca_dim <= min(len(real) - 1, real.shape[1]):
            raise ValueError(f"pca_dim must be in [1, {min(len(real) - 1, real.shape[1])}]")
        self.n_real = len(real)
        self.pca_dim = pca_dim
        self.mu = real.mean(axis=0)
        _, _, vt = np.linalg.svd(real - self.mu, full_matrices=False)
        self.basis = vt[:pca_dim].T
        self.sigma_pca = _cov((real - self.mu) @ self.basis)
        self.sigma_full = _cov(real)
        self.norm_pca = float(np.linalg.norm(self.sigma_pca))
        self.norm_full = float(np.linalg.norm(self.sigma_full))

    @classmethod
    def try_create(cls, pca_dim: int = 64):
        if not (Path(config.SYNTH_EMBEDDINGS_PT).is_file() and Path(config.REAL_TRAIN_EMBEDDINGS_PT).is_file()):
            return None
        try:
            return cls(pca_dim=pca_dim)
        except Exception as exc:  # torch missing, corrupt file, ...
            print(f"[diagnostics] covariance diagnostic disabled: {exc}")
            return None

    def _rows(self, source_row_idx) -> np.ndarray:
        idx = np.asarray(source_row_idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self.synth)):
            raise IndexError(f"_source_row_idx out of range for {len(self.synth)} synthetic embeddings")
        return self.synth[idx]

    def distances(self, source_row_idx) -> dict:
        """Both relative distances for the synthetic rows with these _source_row_idx values."""
        x = self._rows(source_row_idx)
        if len(x) < 2:
            return {"n": int(len(x)), "rel_dist_pca": float("nan"), "rel_dist_full": float("nan")}
        x = x.astype(np.float64)
        d_pca = np.linalg.norm(_cov((x - self.mu) @ self.basis) - self.sigma_pca) / self.norm_pca
        d_full = np.linalg.norm(_cov(x) - self.sigma_full) / self.norm_full
        return {"n": int(len(x)), "rel_dist_pca": float(d_pca), "rel_dist_full": float(d_full)}

    def relative_distance(self, source_row_idx) -> float:
        """PCA-space relative distance (used by scripts/02_select_subsets.py)."""
        return self.distances(source_row_idx)["rel_dist_pca"]
