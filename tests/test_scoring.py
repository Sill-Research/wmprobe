import numpy as np
import torch

from wmprobe.scoring import S_GRID, score_from_logits


def _random_case(T=30, V=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(T, V, generator=g) * 3
    tokens = torch.randint(0, V, (T,), generator=g)
    prev = torch.randint(0, V, (T,), generator=g)
    return logits, tokens, prev


def test_score_from_logits_matches_direct_computation():
    logits, tokens, prev = _random_case()
    st = score_from_logits(logits, tokens, prev, k=10)
    logp = torch.log_softmax(logits.double(), -1)
    p = logp.exp()
    np.testing.assert_allclose(st.logp_y, logp[torch.arange(30), tokens].numpy(), atol=1e-5)
    np.testing.assert_allclose(st.entropy, -(p * logp).sum(-1).numpy(), atol=1e-5)
    var = (p * logp**2).sum(-1) - ((p * logp).sum(-1)) ** 2
    np.testing.assert_allclose(st.var_logp, var.numpy(), atol=1e-4)
    np.testing.assert_allclose(st.logit_y, logits[torch.arange(30), tokens].numpy(), atol=1e-6)
    top = torch.topk(logits, 10, dim=-1)
    np.testing.assert_array_equal(st.topk_idx, top.indices.numpy())
    np.testing.assert_array_equal(st.tokens, tokens.numpy())
    np.testing.assert_array_equal(st.prev, prev.numpy())
    assert st.lse.shape == (30, len(S_GRID))


def test_lse_interpolation_is_accurate_off_grid():
    """Operating range: fitted s in [0.5, 2] (tau in [0.5, 2]) and 2s for collisions."""
    logits, tokens, prev = _random_case(seed=1)
    st = score_from_logits(logits, tokens, prev, k=10)
    for s in (0.53, 0.77, 1.0, 1.43, 2.91, 3.77):
        exact = torch.logsumexp(s * logits.double(), -1).numpy()
        np.testing.assert_allclose(st.lse_at(s), exact, atol=1e-4)


def test_topk_probs_match_tempered_softmax():
    logits, tokens, prev = _random_case(seed=2)
    st = score_from_logits(logits, tokens, prev, k=10)
    s = 0.8
    full = torch.softmax(s * logits.double(), -1).numpy()
    np.testing.assert_allclose(
        st.topk_probs(s), np.take_along_axis(full, st.topk_idx, 1), atol=1e-5
    )


def test_excluded_ids_are_scored_as_impossible():
    """Interior tokens of an observed text are never EOS, so detectors condition on that."""
    logits, tokens, prev = _random_case(seed=3)
    tokens[tokens == 7] = 8
    st = score_from_logits(logits, tokens, prev, k=10, exclude_ids=(7,))
    masked = logits.clone()
    masked[:, 7] = float("-inf")
    ref = score_from_logits(masked, tokens, prev, k=10)
    np.testing.assert_allclose(st.logp_y, ref.logp_y, atol=1e-6)
    np.testing.assert_allclose(st.entropy, ref.entropy, atol=1e-6)
    np.testing.assert_allclose(st.lse, ref.lse, atol=1e-6)
    assert np.isfinite(st.var_logp).all() and not (st.topk_idx == 7).any()
