import numpy as np
import pytest

from wmprobe.stats import roc_auc, roc_curve


def test_auc_perfect_random_and_ties():
    assert roc_auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert roc_auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert roc_auc([1, 1], [1, 1]) == 0.5
    assert roc_auc([2, 0], [1]) == 0.5


def test_auc_matches_large_sample_gaussian():
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 20000), rng.normal(0, 1, 20000)
    # AUC = Phi(1 / sqrt(2)) ~= 0.760
    assert roc_auc(pos, neg) == pytest.approx(0.760, abs=0.01)


def test_roc_curve_endpoints_and_monotone():
    fpr, tpr = roc_curve([3, 1, 4], [0, 2, 1])
    assert (fpr[0], tpr[0]) == (0.0, 0.0) and (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
