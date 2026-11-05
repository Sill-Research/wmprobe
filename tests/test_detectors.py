"""Detector calibration (H0) and power (H1) on the exact toy model.

Each test fixes its seeds, so outcomes are deterministic; thresholds are loose
enough (|z| < 3 under H0) that a correct implementation passes for any seed with
high probability.
"""

import numpy as np
import pytest
import torch
from toy import ToyLM, V, corpus, random_prompts, sample, score

from wmprobe.detectors import (
    bigram_residual_test,
    context_interaction_test,
    diversity_collapse_test,
    fit_inverse_temperature,
    likelihood_deficit_z,
    select_answer_set,
)
from wmprobe.watermarks import GumbelWatermark, KGWWatermark, PlainSampler, TournamentWatermark

LM = ToyLM()
KGW = KGWWatermark(key=99, gamma=0.25, delta=2.0, vocab_size=V)
SYNTHID = TournamentWatermark(key=11, depth=30, vocab_size=V)
SYNTHID_N4 = TournamentWatermark(key=11, depth=30, vocab_size=V, n_competitors=4)


@pytest.fixture(scope="module")
def corpora():
    return {
        "t1.0": corpus(LM, PlainSampler(1.0), 120, 150, seed=1),
        "t0.7": corpus(LM, PlainSampler(0.7), 120, 150, seed=2),
        "t1.3": corpus(LM, PlainSampler(1.3), 120, 150, seed=3),
        "kgw": corpus(LM, KGW, 120, 150, seed=4),
        "gumbel_h4": corpus(LM, GumbelWatermark(5, 4, V), 120, 150, seed=5),
        "gumbel_h8": corpus(LM, GumbelWatermark(5, 8, V), 120, 150, seed=7),
        "synthid": corpus(LM, SYNTHID, 120, 150, seed=8),
        "synthid_n4": corpus(LM, SYNTHID_N4, 120, 150, seed=9),
    }


# ---------------------------------------------------------------- temperature fit


@pytest.mark.parametrize("name,tau", [("t1.0", 1.0), ("t0.7", 0.7), ("t1.3", 1.3)])
def test_fit_inverse_temperature_recovers_tau(corpora, name, tau):
    assert abs(1 / fit_inverse_temperature(corpora[name]) - tau) < 0.04


# ---------------------------------------------------------------- Test A


def test_a_calibrated_under_h0(corpora):
    z = np.array([likelihood_deficit_z(t) for t in corpora["t1.0"]])
    assert abs(z.mean()) < 0.3
    assert 0.75 < z.std() < 1.25


def test_a_detects_kgw(corpora):
    """Weak per text: on the toy, E[S_t] = -0.16 nats vs sd 1.68, so E[Z] ~ -1.2 at T=150."""
    z = np.array([likelihood_deficit_z(t) for t in corpora["kgw"]])
    assert z.mean() < -0.7


def test_a_is_confounded_by_temperature(corpora):
    """The documented confound: plain T=1.3 also yields strongly negative Z."""
    z = np.array([likelihood_deficit_z(t) for t in corpora["t1.3"]])
    assert z.mean() < -3


def test_a_blind_to_gumbel(corpora):
    z = np.array([likelihood_deficit_z(t) for t in corpora["gumbel_h4"]])
    assert abs(z.mean()) < 0.4


def test_a_blind_to_non_distortionary_synthid(corpora):
    """N=2 + repeated-context masking is single-sequence non-distortionary: E_key[text] = p."""
    z = np.array([likelihood_deficit_z(t) for t in corpora["synthid"]])
    assert abs(z.mean()) < 0.4


def test_a_detects_distortionary_synthid(corpora):
    """N=4 flattens the key-averaged distribution, the same signature as KGW (and heat)."""
    z = np.array([likelihood_deficit_z(t) for t in corpora["synthid_n4"]])
    assert z.mean() < -1.0


def test_a_max_len_truncates(corpora):
    t = corpora["kgw"][0]
    assert likelihood_deficit_z(t, max_len=10) == pytest.approx(
        likelihood_deficit_z(type(t)(**{f: getattr(t, f)[:10] for f in t.__dataclass_fields__}))
    )


# ---------------------------------------------------------------- Test B


@pytest.mark.parametrize("name", ["t1.0", "t0.7", "t1.3"])
def test_b_calibrated_under_h0_any_temperature(corpora, name):
    assert abs(bigram_residual_test(corpora[name]).z) < 3


def test_b_detects_kgw(corpora):
    assert bigram_residual_test(corpora["kgw"]).z > 6


def test_b_blind_to_gumbel_when_contexts_do_not_repeat(corpora):
    """With h=8 no context window recurs across texts, so Gumbel is distortion-free here.

    (With h=4 the toy's 2-token memory makes generation a deterministic finite-state map:
    texts that share one 4-gram merge forever, a genuine repeated-context leak.)
    """
    assert abs(bigram_residual_test(corpora["gumbel_h8"]).z) < 3


def test_b_detects_gumbel_with_one_token_context():
    """h=1: the Gumbel noise is fixed per previous token, so preferences repeat."""
    texts = corpus(LM, GumbelWatermark(5, 1, V), 120, 150, seed=6)
    assert bigram_residual_test(texts).z > 6


def _one_cell_corpus(seed: int, n_texts: int = 30, n_pos: int = 20):
    """Exact H0 where a single (prev, next) cell carries all the variance (like '.'->'\\n')."""
    g = torch.Generator().manual_seed(seed)
    texts = []
    for _ in range(n_texts):
        tokens = torch.randint(0, 2, (n_pos,), generator=g)
        texts.append(score(_TwoTokenLM(), torch.cat([torch.zeros(2, dtype=torch.long), tokens]), 2))
    return texts


class _TwoTokenLM:
    def teacher_logits(self, ids):
        return torch.zeros(len(ids) - 2, 2)


def test_b_sign_flip_p_value_is_calibrated_when_one_cell_dominates():
    res = [bigram_residual_test(_one_cell_corpus(i), s=1.0) for i in range(400)]
    p_rate = np.mean([r.p_value < 0.01 for r in res])
    z_rate = np.mean([r.z > 2.326 for r in res])
    assert p_rate < 0.03
    assert z_rate > 0.02  # why the normal approximation alone is not enough


@pytest.mark.parametrize("name", ["t1.0", "t0.7", "t1.3"])
def test_b_p_value_not_small_under_h0(corpora, name):
    assert bigram_residual_test(corpora[name]).p_value > 0.01


def test_b_p_value_small_for_kgw(corpora):
    assert bigram_residual_test(corpora["kgw"]).p_value < 0.01


def test_b_reports_fitted_temperature(corpora):
    res = bigram_residual_test(corpora["t1.3"])
    assert abs(1 / res.s - 1.3) < 0.04


# ---------------------------------------------------------------- Test C


def _interaction_case(kind: str, n: int = 2000, k: int = 8, seed: int = 0):
    """Simulate n one-token queries per context variant; variant c ends in token c.

    Variants differ only in a neutral separator token, so they share a base answer
    distribution plus a small variant-specific perturbation.
    """
    g = torch.Generator().manual_seed(seed)
    proxy = torch.randn(V, generator=g) * 2 + torch.randn(k, V, generator=g) * 0.3
    if kind == "plain":
        gen = proxy / 0.8
    elif kind == "word_effect":  # proxy misses a context-independent per-word bias
        gen = proxy + torch.randn(V, generator=g)
    elif kind == "kgw":
        ids = torch.arange(k).unsqueeze(1)
        gen = KGW(ids, proxy.clone())
    else:
        raise ValueError(kind)
    probs = torch.softmax(gen, -1)
    counts_full = torch.stack(
        [torch.bincount(torch.multinomial(p, n, True, generator=g), minlength=V) for p in probs]
    ).numpy()
    w = select_answer_set(proxy.numpy(), n)
    return counts_full[:, w], proxy.numpy()[:, w]


def test_select_answer_set_respects_expected_count():
    proxy = np.log(np.array([[0.5, 0.3, 0.15, 0.05], [0.5, 0.3, 0.15, 0.05]]))
    w = select_answer_set(proxy, n_queries=40, min_expected=5, max_size=10)
    assert sorted(w.tolist()) == [0, 1, 2]
    assert len(select_answer_set(proxy, 1000, max_size=2)) == 2


def test_select_answer_set_requires_count_in_every_variant():
    """The chi^2 reference needs every cell's expected count large, not just the average."""
    proxy = np.log(np.array([[0.5, 0.3, 0.15, 0.05], [0.5, 0.02, 0.15, 0.33]]))
    w = select_answer_set(proxy, n_queries=40, min_expected=5, max_size=10)
    assert sorted(w.tolist()) == [0, 2]


@pytest.mark.parametrize("kind", ["plain", "word_effect"])
def test_c_g2_matches_chi2_reference_under_h0(kind):
    res = [context_interaction_test(*_interaction_case(kind, n=500, seed=i)) for i in range(60)]
    assert 0.85 < np.mean([r.g2 / r.df for r in res]) < 1.15
    assert np.mean([r.p_value < 0.05 for r in res]) < 0.12


@pytest.mark.parametrize("kind", ["plain", "word_effect"])
def test_c_calibrated_under_h0(kind):
    assert context_interaction_test(*_interaction_case(kind)).p_value > 0.01


def test_c_recovers_temperature():
    res = context_interaction_test(*_interaction_case("plain"))
    assert abs(res.s - 1 / 0.8) < 0.1


def test_c_detects_kgw():
    assert context_interaction_test(*_interaction_case("kgw")).p_value < 1e-6


# ---------------------------------------------------------------- Test D


def _repeats(processor, n_prompts=20, n_rep=10, n_new=30, seed=0):
    """n_rep samples of each prompt -> disjoint pairs of ScoredTexts (prompt visible)."""
    prompts = random_prompts(n_prompts, seed=seed)
    pairs, everything = [], []
    for i, prompt in enumerate(prompts):
        batch = prompt.unsqueeze(0).repeat(n_rep, 1)
        ids = sample(LM, batch, n_new, processor, seed=seed + 100 + i)
        scored = [score(LM, row, prompts.shape[1]) for row in ids]
        everything += scored
        pairs += list(zip(scored[0::2], scored[1::2], strict=True))
    return pairs, everything


@pytest.mark.parametrize("tau", [0.7, 1.0, 1.3])
def test_d_calibrated_under_h0(tau):
    pairs, texts = _repeats(PlainSampler(tau))
    res = diversity_collapse_test(pairs, fit_inverse_temperature(texts))
    assert abs(res.z) < 3


def test_d_detects_gumbel():
    pairs, texts = _repeats(GumbelWatermark(5, 4, V))
    res = diversity_collapse_test(pairs, fit_inverse_temperature(texts))
    assert res.z > 10
    assert res.observed > res.expected


def test_d_detects_synthid():
    """A fixed key concentrates each next-token distribution, so repeats agree too often."""
    pairs, texts = _repeats(SYNTHID)
    res = diversity_collapse_test(pairs, fit_inverse_temperature(texts))
    assert res.z > 5
    assert abs(1 / fit_inverse_temperature(texts) - 1.0) < 0.1  # marginally still T=1


def test_d_does_not_flag_kgw():
    pairs, texts = _repeats(KGW)
    assert abs(diversity_collapse_test(pairs, fit_inverse_temperature(texts)).z) < 3


@pytest.mark.parametrize("counts", [np.zeros((3, 0)), np.array([[5, 0], [0, 0], [0, 0]])])
def test_c_rejects_degenerate_tables(counts):
    with pytest.raises(ValueError):
        context_interaction_test(counts, np.zeros(counts.shape))
