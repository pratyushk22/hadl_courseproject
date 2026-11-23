import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hadl_eval.metrics import (bootstrap_ci, char_f1, contains_gold, exact_match, loo_calibrated_error,  # noqa: E402
                               paired_bootstrap_test, proxy_error, score_example, token_f1)
from hadl_eval.text_utils import clean_answer_target, clean_generation, normalize_answer  # noqa: E402


def test_normalize_odia():
    assert normalize_answer("ଭୁବନେଶ୍ୱର।") == "ଭୁବନେଶ୍ୱର"
    assert normalize_answer("  ୨୦୧୪  ") == "2014"            # Odia digits -> ASCII
    assert normalize_answer("ଇନ୍‌ସଫିସିଏନ୍‌ସି") == "ଇନ୍ସଫିସିଏନ୍ସି"  # ZWNJ removed
    assert normalize_answer("A, b;  C!") == "a b c"
    assert normalize_answer(None) == ""
    # vowel signs / virama are marks (not punctuation) and must survive
    assert normalize_answer("ରାଜ୍ୟ") == "ରାଜ୍ୟ"


def test_em_f1():
    assert exact_match("ଭୁବନେଶ୍ୱର ।", "ଭୁବନେଶ୍ୱର") == 1.0
    assert token_f1("ରାଜା ନରସିଂହ ଦେବ", "ନରସିଂହ ଦେବ") == pytest.approx(0.8)
    assert token_f1("", "x") == 0.0
    assert token_f1("", "") == 1.0
    assert char_f1("ଭୁବନେଶ୍ୱରରେ", "ଭୁବନେଶ୍ୱର") > 0.8          # suffix-tolerant
    assert contains_gold("ଏହା 2014 ମସିହା", "୨୦୧୪") == 1.0


def test_score_example_takes_max_over_golds():
    s = score_example("ଦିଲ୍ଲୀପ", ["ଦିଲ୍ଲୀପ ରାୟ", "ଦିଲ୍ଲୀପ"])
    assert s["exact_match"] == 1.0 and s["token_f1"] == 1.0
    assert score_example("x", [])["exact_match"] == 0.0


def test_clean_targets_and_generations():
    assert clean_answer_target("ପ୍ରାଇମେରୀ ଆଡ୍ରେନାଲ ଇନ୍‌ସଫିସିଏନ୍‌ସି।") == "ପ୍ରାଇମେରୀ ଆଡ୍ରେନାଲ ଇନ୍‌ସଫିସିଏନ୍‌ସି"
    assert clean_generation("<think>\n\n</think>\n\nଭୁବନେଶ୍ୱର।\nextra") == "ଭୁବନେଶ୍ୱର"
    assert clean_generation("Answer: ଦିଲ୍ଲୀପ") == "ଦିଲ୍ଲୀପ"
    assert clean_generation("") == ""


def test_bootstrap_ci_and_clusters():
    v = np.r_[np.ones(50), np.zeros(50)]
    mean, lo, hi = bootstrap_ci(v, n_boot=500, seed=1)
    assert mean == 0.5 and lo < 0.5 < hi
    groups = np.repeat(np.arange(10), 10)
    m2, lo2, hi2 = bootstrap_ci(v, groups=groups, n_boot=500, seed=1)
    assert m2 == 0.5 and (hi2 - lo2) > (hi - lo)     # clustered CI is wider for clustered data


def test_paired_test():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    a = b + 0.2
    res = paired_bootstrap_test(a, b, n_boot=1000)
    assert res["diff"] == pytest.approx(0.2) and res["p_value"] < 0.01
    same = paired_bootstrap_test(b, b, n_boot=200)
    assert same["diff"] == 0.0 and same["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_test([1, 2], [1])


def test_proxy_error():
    e = proxy_error([110, 90], [100, 100])
    assert e["mape_pct"] == pytest.approx(10.0)
    assert e["mean_signed_error_pct"] == pytest.approx(0.0)
    assert e["scale_actual_over_pred"] == pytest.approx(1.0)
    scaled = proxy_error([200, 400, 600], [100, 200, 300])   # constant factor off, perfect ranking
    assert scaled["spearman"] == pytest.approx(1.0)
    assert loo_calibrated_error([200, 400, 600], [100, 200, 300])["loo_calibrated_mape_pct"] == pytest.approx(0.0)
