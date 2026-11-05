from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie), via Mann-Whitney ranks."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    ranks = rankdata(np.concatenate([pos, neg]))
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def roc_curve(pos: Sequence[float], neg: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = np.array([0.0] + [(pos >= t).mean() for t in thresholds])
    fpr = np.array([0.0] + [(neg >= t).mean() for t in thresholds])
    return fpr, tpr
