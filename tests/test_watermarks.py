import numpy as np
import pytest
import torch
from scipy import stats
from toy import ToyLM, V, random_prompts, sample

from wmprobe.watermarks import (
    GumbelWatermark,
    KGWWatermark,
    PlainSampler,
    TournamentWatermark,
    gumbel_z_score,
    kgw_z_score,
    prf_seed,
    tournament_z_score,
)


def test_prf_seed_is_deterministic_and_key_dependent():
    assert prf_seed(1, 5, 6) == prf_seed(1, 5, 6)
    assert prf_seed(1, 5, 6) != prf_seed(2, 5, 6)
    assert prf_seed(1, 5, 6) != prf_seed(1, 6, 5)
    assert 0 <= prf_seed(1, 5) < 2**63


def test_plain_sampler_applies_temperature_and_suppression():
    proc = PlainSampler(temperature=2.0, suppress_ids=(3,))
    out = proc(torch.zeros(1, 2, dtype=torch.long), torch.full((1, 5), 4.0))
    assert torch.allclose(out[0, [0, 1, 2, 4]], torch.full((4,), 2.0))
    assert out[0, 3] == float("-inf")


def test_kgw_green_mask_size_and_determinism():
    wm = KGWWatermark(key=7, gamma=0.25, delta=2.0, vocab_size=V)
    m1, m2 = wm.green_mask(11), wm.green_mask(11)
    assert m1.dtype == torch.bool and m1.shape == (V,)
    assert int(m1.sum()) == int(0.25 * V)
    assert torch.equal(m1, m2)
    assert not torch.equal(m1, wm.green_mask(12))


def test_kgw_processor_adds_delta_to_green_only():
    wm = KGWWatermark(key=7, gamma=0.5, delta=2.0, vocab_size=V)
    ids = torch.tensor([[1, 2, 9]])
    out = wm(ids, torch.zeros(1, V))
    green = wm.green_mask(9)
    assert torch.allclose(out[0, green], torch.full((int(green.sum()),), 2.0))
    assert torch.allclose(out[0, ~green], torch.zeros(int((~green).sum())))


def test_gumbel_processor_is_one_hot_and_deterministic():
    wm = GumbelWatermark(key=3, context_width=2, vocab_size=V)
    ids = torch.tensor([[4, 5, 6], [9, 5, 6]])  # same last-2 context
    logits = torch.randn(1, V).repeat(2, 1)
    out = wm(ids, logits)
    assert torch.isfinite(out).sum(dim=1).tolist() == [1, 1]
    assert out[0].argmax() == out[1].argmax()


def test_gumbel_sampling_is_distortion_free_over_keys():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    p = torch.softmax(logits.double(), -1)[0].numpy()
    ids = torch.tensor([[0, 1]])
    picks = [
        int(GumbelWatermark(key=k, context_width=1, vocab_size=5)(ids, logits).argmax())
        for k in range(20000)
    ]
    counts = np.bincount(picks, minlength=5)
    assert stats.chisquare(counts, p * counts.sum()).pvalue > 1e-3


def _toy_ids(processor, n=8, n_new=150, seed=0):
    lm = ToyLM()
    prompts = random_prompts(n, seed=seed)
    return sample(lm, prompts, n_new, processor, seed=seed), prompts.shape[1]


def test_kgw_keyed_detector_separates_watermarked_from_plain():
    wm = KGWWatermark(key=42, gamma=0.25, delta=2.0, vocab_size=V)
    marked, start = _toy_ids(wm)
    plain, _ = _toy_ids(PlainSampler())
    assert min(kgw_z_score(row.tolist(), start, wm) for row in marked) > 4
    assert max(abs(kgw_z_score(row.tolist(), start, wm)) for row in plain) < 3.5


def test_gumbel_keyed_detector_separates_watermarked_from_plain():
    wm = GumbelWatermark(key=42, context_width=2, vocab_size=V)
    marked, start = _toy_ids(wm)
    plain, _ = _toy_ids(PlainSampler())
    assert min(gumbel_z_score(row.tolist(), start, wm) for row in marked) > 4
    assert max(abs(gumbel_z_score(row.tolist(), start, wm)) for row in plain) < 3.5


# ---------------------------------------------------------------- Tournament (SynthID-Text)


def test_tournament_g_values_are_deterministic_balanced_bits():
    wm = TournamentWatermark(key=7, depth=30, vocab_size=V)
    g = wm.g_values([1, 2, 3, 4])
    assert g.dtype == bool and g.shape == (30, V)
    assert np.array_equal(g, wm.g_values([1, 2, 3, 4]))
    assert not np.array_equal(g, wm.g_values([1, 2, 3, 5]))
    assert not np.array_equal(g, TournamentWatermark(8, 30, V).g_values([1, 2, 3, 4]))
    assert abs(g.mean() - 0.5) < 0.03
    assert len({row.tobytes() for row in g}) == 30  # every layer has its own g-function
    assert np.array_equal(wm.g_values([1, 2, 3, 4], np.array([5, 0, 9])), g[:, [5, 0, 9]])


def _explicit_tournament(p, g, n, trials, rng):
    """Algorithm 2 of the paper, literally: draw n**m candidates, knock out in matches of n.

    Ties are broken uniformly by adding noise in [0, 0.5) to the 0/1 g-values.
    """
    m = len(g)
    ys = rng.choice(len(p), size=(trials, n**m), p=p)
    for layer in range(m):
        ys = ys.reshape(trials, -1, n)
        score = g[layer][ys] + rng.random(ys.shape) * 0.5
        ys = np.take_along_axis(ys, score.argmax(axis=2)[..., None], axis=2)[..., 0]
    return ys[:, 0]


@pytest.mark.parametrize("n", [2, 3])
def test_tournament_distribution_matches_explicit_algorithm(n):
    """The processor's closed form is exactly the output law of the knockout tournament."""
    wm = TournamentWatermark(key=1, depth=3, vocab_size=6, context_width=2, n_competitors=n)
    ids = torch.tensor([[3, 4]])
    logits = torch.tensor([[1.5, 1.0, 0.3, 0.0, -0.5, -1.0]])
    q = torch.softmax(wm(ids, logits.clone()).double(), -1)[0].numpy()
    p = torch.softmax(logits.double(), -1)[0].numpy()
    assert np.abs(q - p).max() > 0.02  # the watermark actually moved the distribution
    trials = 40_000
    draws = _explicit_tournament(p, wm.g_values([3, 4]), n, trials, np.random.default_rng(0))
    counts = np.bincount(draws, minlength=6)
    assert stats.chisquare(counts, q * trials).pvalue > 1e-3


def _key_averaged(n: int, logits: torch.Tensor, n_seeds: int = 20_000) -> np.ndarray:
    """E over random seeds of the sampling distribution (rows = distinct 2-token contexts)."""
    wm = TournamentWatermark(
        key=0, depth=30, vocab_size=logits.shape[1], context_width=2, n_competitors=n
    )
    ids = torch.stack([torch.arange(n_seeds), torch.arange(n_seeds) + n_seeds], dim=1)
    q = torch.softmax(wm(ids, logits.repeat(n_seeds, 1)).double(), -1)
    return q.mean(0).numpy()


def test_tournament_with_two_competitors_is_distortion_free_over_keys():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    p = torch.softmax(logits.double(), -1)[0].numpy()
    assert np.abs(_key_averaged(2, logits) - p).max() < 0.015


def test_tournament_with_more_competitors_flattens_on_average():
    """N > 2 is the distortionary mode: like KGW, the key-averaged distribution is flatter."""
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    p = torch.softmax(logits.double(), -1)[0].numpy()
    qbar = _key_averaged(4, logits)
    assert qbar[0] < p[0] - 0.05
    assert stats.entropy(qbar) > stats.entropy(p) + 0.1


def test_tournament_skips_repeated_context_windows():
    """Repeated context masking: a window already seen in the sequence is sampled plainly."""
    wm = TournamentWatermark(key=7, depth=30, vocab_size=V, context_width=2)
    logits = torch.randn(2, V, generator=torch.Generator().manual_seed(0)) * 2
    ids = torch.tensor([[1, 2, 3, 1, 2], [1, 2, 3, 4, 2]])
    out = torch.log_softmax(wm(ids, logits.clone()), -1)
    plain = torch.log_softmax(logits, -1)
    assert torch.allclose(out[0], plain[0], atol=1e-5)
    assert not torch.allclose(out[1], plain[1], atol=1e-2)


@pytest.mark.parametrize("n", [2, 4])
def test_tournament_keyed_detector_separates_watermarked_from_plain(n):
    wm = TournamentWatermark(key=42, depth=30, vocab_size=V, n_competitors=n)
    marked, start = _toy_ids(wm)
    plain, _ = _toy_ids(PlainSampler())
    assert min(tournament_z_score(row.tolist(), start, wm) for row in marked) > 4
    assert max(abs(tournament_z_score(row.tolist(), start, wm)) for row in plain) < 3.5


def test_tournament_keyed_detector_is_standard_normal_under_h0():
    """g-values of distinct contexts are iid Bernoulli(1/2) whatever produced the text."""
    wm = TournamentWatermark(key=42, depth=30, vocab_size=V)
    plain, start = _toy_ids(PlainSampler(1.3), n=200, n_new=100, seed=1)
    z = np.array([tournament_z_score(row.tolist(), start, wm) for row in plain])
    assert abs(z.mean()) < 0.25
    assert 0.8 < z.std() < 1.2


def test_tournament_keyed_detector_ignores_repeated_contexts():
    """After `a` repeats, every context has been seen: only the two straddling ones count."""
    wm = TournamentWatermark(key=42, depth=30, vocab_size=V, context_width=2)
    a = list(range(50))
    assert tournament_z_score(a + a, 0, wm) == pytest.approx(tournament_z_score(a + a[:2], 0, wm))
