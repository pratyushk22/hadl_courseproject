import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadl_eval.diagnostics import CovarianceDiagnostic  # noqa: E402


def test_matched_subset_is_closer_than_shifted_subset():
    rng = np.random.default_rng(0)
    d = 16
    real = rng.normal(size=(400, d))
    synth = np.vstack([rng.normal(size=(500, d)),                       # same distribution as real
                       rng.normal(size=(500, d)) * np.linspace(0.2, 3, d)])  # covariance-shifted
    diag = CovarianceDiagnostic(pca_dim=8, synth=synth, real=real)
    near = diag.distances(np.arange(500))
    far = diag.distances(np.arange(500, 1000))
    assert near["n"] == 500
    assert near["rel_dist_pca"] < far["rel_dist_pca"]
    assert near["rel_dist_full"] < far["rel_dist_full"]
    # mean shift does not change a covariance distance
    shifted = CovarianceDiagnostic(pca_dim=8, synth=synth + 5.0, real=real)
    assert shifted.distances(np.arange(500))["rel_dist_full"] == pytest.approx(near["rel_dist_full"])
    assert np.isnan(diag.distances([3])["rel_dist_pca"])
    with pytest.raises(IndexError):
        diag.distances([5000])
    assert diag.relative_distance(np.arange(500)) == pytest.approx(near["rel_dist_pca"])
