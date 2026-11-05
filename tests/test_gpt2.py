"""Integration with real GPT-2 weights (cached by Hugging Face after the first run)."""

import numpy as np
import pytest
import torch

from wmprobe.generation import generate, load, next_token_probs
from wmprobe.scoring import score_continuations
from wmprobe.watermarks import (
    GumbelWatermark,
    KGWWatermark,
    PlainSampler,
    TournamentWatermark,
    kgw_z_score,
    tournament_z_score,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def gpt2():
    return load("gpt2")


@pytest.fixture(scope="module")
def prompts(gpt2):
    _, tok = gpt2
    texts = ["The history of the city begins", "My favourite recipe for soup is", "In 1998, the"]
    return [tok.encode(t) for t in texts]


def test_generate_returns_exact_lengths_without_eos(gpt2, prompts):
    model, tok = gpt2
    out = generate(model, tok, prompts, PlainSampler(suppress_ids=[tok.eos_token_id]), 40, seed=0)
    assert [len(o) for o in out] == [40, 40, 40]
    assert all(tok.eos_token_id not in o for o in out)


def test_generate_is_reproducible_given_seed(gpt2, prompts):
    model, tok = gpt2
    proc = PlainSampler()
    assert generate(model, tok, prompts, proc, 20, seed=3) == generate(
        model, tok, prompts, proc, 20, seed=3
    )


def test_kgw_generation_is_detected_by_keyed_detector(gpt2, prompts):
    model, tok = gpt2
    wm = KGWWatermark(1, 0.25, 2.0, model.config.vocab_size)
    out = generate(model, tok, prompts, wm, 100, seed=0)
    z = [kgw_z_score(p + o, len(p), wm) for p, o in zip(prompts, out, strict=True)]
    plain = generate(model, tok, prompts, PlainSampler(), 100, seed=0)
    z0 = [kgw_z_score(p + o, len(p), wm) for p, o in zip(prompts, plain, strict=True)]
    assert min(z) > 4 and max(map(abs, z0)) < 3


def test_synthid_generation_is_detected_by_keyed_detector(gpt2, prompts):
    model, tok = gpt2
    wm = TournamentWatermark(1, 30, model.config.vocab_size, suppress_ids=[tok.eos_token_id])
    out = generate(model, tok, prompts, wm, 100, seed=0)
    z = [tournament_z_score(p + o, len(p), wm) for p, o in zip(prompts, out, strict=True)]
    plain = generate(model, tok, prompts, PlainSampler(), 100, seed=0)
    z0 = [tournament_z_score(p + o, len(p), wm) for p, o in zip(prompts, plain, strict=True)]
    assert min(z) > 4 and max(map(abs, z0)) < 3


def test_gumbel_generation_is_deterministic_across_seeds(gpt2, prompts):
    model, tok = gpt2
    wm = GumbelWatermark(1, 4, model.config.vocab_size)
    assert generate(model, tok, prompts, wm, 30, seed=0) == generate(
        model, tok, prompts, wm, 30, seed=1
    )


def test_batched_scoring_matches_unbatched(gpt2, prompts):
    model, _ = gpt2
    conts = [list(range(100, 100 + n)) for n in (5, 17, 9)]
    batched = score_continuations(model, prompts, conts, batch_size=3)
    for p, c, b in zip(prompts, conts, batched, strict=True):
        single = score_continuations(model, [p], [c], batch_size=1)[0]
        np.testing.assert_allclose(b.logp_y, single.logp_y, atol=1e-4)
        np.testing.assert_allclose(b.lse, single.lse, atol=1e-3)
    ids = torch.tensor([prompts[1] + conts[1]])
    with torch.no_grad():
        direct = torch.log_softmax(model(ids).logits[0], -1)
    n = len(prompts[1])
    expect = direct[torch.arange(n - 1, n - 1 + 17), torch.tensor(conts[1])].numpy()
    np.testing.assert_allclose(batched[1].logp_y, expect, atol=1e-4)


def test_next_token_probs_applies_processor(gpt2, prompts):
    model, _ = gpt2
    q = next_token_probs(model, prompts, GumbelWatermark(1, 4, model.config.vocab_size))
    assert q.shape == (3, model.config.vocab_size)
    assert torch.allclose(q.max(dim=1).values, torch.ones(3))
    p = next_token_probs(model, prompts, PlainSampler())
    torch.testing.assert_close(p.sum(dim=1), torch.ones(3))
