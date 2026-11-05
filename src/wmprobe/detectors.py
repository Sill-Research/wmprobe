"""Keyless watermark tests. All operate on proxy-scored text (`ScoredText`).

A  likelihood_deficit_z       one text: are tokens less likely than the proxy expects?
B  bigram_residual_test       corpus: do proxy residuals share a previous-token-keyed pattern
                              across independent texts (after fitting out temperature)?
C  context_interaction_test   queries: does the token before the answer change the answer
                              distribution in a way temperature + per-word bias can't explain?
D  diversity_collapse_test    queries: do repeated samples agree more than their entropy allows?
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_softmax, logsumexp, softmax
from scipy.stats import chi2, norm

from wmprobe.scoring import S_GRID, ScoredText


def fit_inverse_temperature(texts: Sequence[ScoredText]) -> float:
    """MLE of s = 1/tau under y_t ~ softmax(s * proxy_logits_t), pooled over texts."""
    sum_logit_y = sum(float(t.logit_y.sum()) for t in texts)
    total_lse = CubicSpline(S_GRID, sum(t.lse.sum(axis=0) for t in texts))
    res = minimize_scalar(
        lambda s: float(total_lse(s)) - s * sum_logit_y,
        bounds=(S_GRID[0], S_GRID[-1]),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(res.x)


# ---------------------------------------------------------------- Test A


def likelihood_deficit_z(text: ScoredText, max_len: int | None = None) -> float:
    """Z = sum(log p(y_t) + H_t) / sqrt(sum Var_t). ~N(0,1) if y ~ proxy at T=1."""
    n = len(text) if max_len is None else min(max_len, len(text))
    centered = text.logp_y[:n] + text.entropy[:n]
    return float(centered.sum() / math.sqrt(text.var_logp[:n].sum()))


# ---------------------------------------------------------------- Test B


@dataclass
class BigramResult:
    z: float  # stat / sqrt(var): effect size, for ranking
    p_value: float  # sign-flip randomization p-value: use this for decisions
    stat: float  # sum over text pairs of <R_j, R_j'>
    var: float  # unbiased H0 variance estimate of stat
    s: float  # inverse temperature the residuals were computed at


def bigram_residual_test(
    texts: Sequence[ScoredText], s: float | None = None, n_flips: int = 2000, seed: int = 0
) -> BigramResult:
    """Cross-text correlation of previous-token-keyed residuals.

    Per text j: R_j(a, b) = sum_{t: y_{t-1}=a} 1[y_t=b, b in top-k_t] - p_t^s(b) 1[b in top-k_t].
    Restricting both terms to the context-determined top-k set keeps each increment an exact
    martingale difference, so E<R_j, R_j'> = 0 for independent texts under H0 (proxy correct
    up to temperature). A watermark that biases tokens as a function of the previous token
    makes residual patterns line up across texts, so the statistic grows ~ n^2.

    The statistic is a quadratic form; when a few high-variance cells dominate (real text:
    '.' -> '\\n') it is chi^2-like and z = stat / sqrt(var) has a heavy right tail. The
    p-value therefore comes from flipping the sign of each (independent, mean-zero) R_j.
    """
    if s is None:
        s = fit_inverse_temperature(texts)
    owners, keys, vals = [], [], []
    for j, t in enumerate(texts):
        p = t.topk_probs(s)
        hit = (t.topk_idx == t.tokens[:, None]).any(axis=1)
        k_exp = (t.prev[:, None] << 20 | t.topk_idx).ravel()
        k_obs = t.prev[hit] << 20 | t.tokens[hit]
        keys += [k_exp, k_obs]
        vals += [-p.ravel(), np.ones(len(k_obs))]
        owners.append(np.full(len(k_exp) + len(k_obs), j))
    uniq, cols = np.unique(np.concatenate(keys), return_inverse=True)
    x = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(owners), cols)), shape=(len(texts), len(uniq))
    )
    gram = (x @ x.T).toarray()
    diag = np.diag(gram)
    stat = (gram.sum() - diag.sum()) / 2
    var = ((gram**2).sum() - (diag**2).sum()) / 2
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_flips, len(texts)))
    flipped = (np.einsum("fi,ij,fj->f", signs, gram, signs) - diag.sum()) / 2
    p_value = (1 + np.sum(flipped >= stat - 1e-9 * abs(stat))) / (1 + n_flips)
    z = float(stat / math.sqrt(var)) if var > 0 else 0.0
    return BigramResult(z=z, p_value=float(p_value), stat=stat, var=var, s=s)


# ---------------------------------------------------------------- Test C


@dataclass
class InteractionResult:
    g2: float
    df: int
    p_value: float
    z: float  # one-sided normal quantile of p_value
    s: float  # fitted inverse temperature under H0


def select_answer_set(
    proxy_logits: np.ndarray, n_queries: int, min_expected: float = 5.0, max_size: int = 20
) -> np.ndarray:
    """Answer tokens whose proxy-expected count is >= min_expected in every variant.

    Every cell must clear the bar for the G^2 statistic to follow its chi^2 reference.
    """
    min_p = softmax(proxy_logits, axis=1).min(axis=0)
    order = np.argsort(-min_p)
    return order[min_p[order] * n_queries >= min_expected][:max_size]


def context_interaction_test(counts: np.ndarray, proxy_logits: np.ndarray) -> InteractionResult:
    """G^2 test of H0: pi_c(w) ∝ exp(s * l_c(w) + beta_w) against a saturated model.

    counts [K, M]: answers observed for each of K prompt variants (differing only in the
    token before the answer), restricted to an answer set of M tokens. proxy_logits [K, M].
    H0 absorbs unknown temperature (s) and context-independent proxy error (beta); a
    red-green watermark adds a pseudorandom variant-by-word term that it cannot absorb.
    """
    keep = counts.sum(axis=1) > 0
    n, logit = counts[keep].astype(float), proxy_logits[keep].astype(float)
    k, m = n.shape
    if k < 2 or m < 2:
        raise ValueError(f"need >= 2 answered variants and >= 2 answers (K={k}, M={m})")
    totals = n.sum(axis=1, keepdims=True)

    def nll(theta):
        eta = theta[0] * logit + np.concatenate([[0.0], theta[1:]])
        log_pi = log_softmax(eta, axis=1)
        resid = n - totals * np.exp(log_pi)
        grad = np.concatenate([[-(resid * logit).sum()], -resid.sum(axis=0)[1:]])
        return -(n * log_pi).sum(), grad

    fit = minimize(nll, np.concatenate([[1.0], np.zeros(m - 1)]), jac=True, method="L-BFGS-B")
    eta = fit.x[0] * logit + np.concatenate([[0.0], fit.x[1:]])
    expected = totals * np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
    pos = n > 0
    g2 = float(2 * (n[pos] * np.log(n[pos] / expected[pos])).sum())
    df = k * (m - 1) - m
    if df <= 0:
        raise ValueError(f"need more variants or answers (K={k}, M={m})")
    p = float(chi2.sf(g2, df))
    return InteractionResult(g2=g2, df=df, p_value=p, z=float(norm.isf(max(p, 1e-300))), s=fit.x[0])


# ---------------------------------------------------------------- Test D


@dataclass
class CollapseResult:
    z: float
    observed: float  # agreements while prefixes were identical
    expected: float  # sum of collision probabilities at those steps
    n_steps: int


def diversity_collapse_test(
    pairs: Sequence[tuple[ScoredText, ScoredText]], s: float
) -> CollapseResult:
    """Do independent samples of the same prompt agree more often than chance?

    Walk each pair while its prefixes are identical; at each step the two next tokens agree
    with probability c_t = sum_w p_t^s(w)^2 under H0. Stop at the first disagreement.
    s should be fitted from the samples' own likelihoods (fit_inverse_temperature), which
    separates "low temperature" from "deterministic at a normal temperature".
    """
    if 2 * s > S_GRID[-1]:
        raise ValueError(f"s={s:.3f} too large for the scoring grid")
    observed = expected = var = 0.0
    n_steps = 0
    for a, b in pairs:
        n = min(len(a), len(b))
        collision = np.clip(np.exp(a.lse_at(2 * s) - 2 * a.lse_at(s)), 0.0, 1.0)[:n]
        for t in range(n):
            agree = a.tokens[t] == b.tokens[t]
            observed += agree
            expected += collision[t]
            var += collision[t] * (1 - collision[t])
            n_steps += 1
            if not agree:
                break
    z = (observed - expected) / math.sqrt(var) if var > 0 else 0.0
    return CollapseResult(z=float(z), observed=observed, expected=expected, n_steps=n_steps)
