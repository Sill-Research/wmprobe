"""Watermarked samplers (as HF LogitsProcessors) and their keyed detectors.

Every processor returns logits whose softmax is the exact sampling distribution, so
they plug into `model.generate(do_sample=True, temperature=1.0, top_k=0)` and into
manual sampling loops alike. Temperature is applied inside the processor so the
watermark always acts on the tempered distribution.
"""

import hashlib
import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
from transformers import LogitsProcessor


def prf_seed(key: int, *tokens: int) -> int:
    """Keyed pseudorandom 63-bit seed for a token context."""
    payload = ",".join(map(str, (key, *tokens))).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


class _Sampler(LogitsProcessor):
    def __init__(self, temperature: float = 1.0, suppress_ids: Sequence[int] = ()):
        self.temperature = temperature
        self.suppress_ids = list(suppress_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.float() / self.temperature
        if self.suppress_ids:
            scores[:, self.suppress_ids] = float("-inf")
        return self._watermark(input_ids, scores)

    def _watermark(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        return scores


class PlainSampler(_Sampler):
    """Ordinary multinomial sampling at a given temperature."""


class KGWWatermark(_Sampler):
    """Red-green watermark (Kirchenbauer et al. 2023), green list seeded by the previous token."""

    def __init__(
        self,
        key: int,
        gamma: float,
        delta: float,
        vocab_size: int,
        temperature: float = 1.0,
        suppress_ids: Sequence[int] = (),
    ):
        super().__init__(temperature, suppress_ids)
        self.key, self.gamma, self.delta, self.vocab_size = key, gamma, delta, vocab_size
        self.green_mask = lru_cache(maxsize=8192)(self._green_mask)

    def _green_mask(self, prev: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(prf_seed(self.key, prev))
        green = torch.randperm(self.vocab_size, generator=g)[: int(self.gamma * self.vocab_size)]
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[green] = True
        return mask

    def _watermark(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        masks = torch.stack([self.green_mask(int(p)) for p in input_ids[:, -1]])
        return scores + self.delta * masks.to(scores.dtype)


class GumbelWatermark(_Sampler):
    """Aaronson's Gumbel-max watermark: y = argmax_w r_w^(1/p_w), r = PRF(last h tokens)."""

    def __init__(
        self,
        key: int,
        context_width: int,
        vocab_size: int,
        temperature: float = 1.0,
        suppress_ids: Sequence[int] = (),
    ):
        super().__init__(temperature, suppress_ids)
        self.key, self.context_width, self.vocab_size = key, context_width, vocab_size

    def uniforms(self, context: Sequence[int]) -> torch.Tensor:
        g = torch.Generator().manual_seed(prf_seed(self.key, *context))
        return torch.rand(self.vocab_size, generator=g, dtype=torch.float64).clamp_(
            1e-12, 1 - 1e-12
        )

    def _watermark(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(scores.double(), dim=-1)
        out = torch.full_like(scores, float("-inf"))
        for i, row in enumerate(input_ids):
            r = self.uniforms(row[-self.context_width :].tolist())
            out[i, int(torch.argmax(torch.log(r) / probs[i]))] = 0.0
        return out


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _splitmix64(z: np.ndarray) -> np.ndarray:
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _context_repeats(row: torch.Tensor, h: int) -> bool:
    """Does the last h-token window of row also occur earlier in row?"""
    if len(row) <= h:
        return False
    windows = row.unfold(0, h, 1)
    return bool((windows[:-1] == windows[-1]).all(dim=1).any())


class TournamentWatermark(_Sampler):
    """SynthID-Text (Dathathri et al. 2024): Tournament sampling, Bernoulli(1/2) g-values.

    The seed is a PRF of the last h tokens; layer l's g-function is bit l of a 64-bit hash of
    (seed, token). The knockout over N**m candidates is computed exactly in distribution: with
    layer-l green set G = {w : g_l(w) = 1}, each layer maps the distribution q to

        q'(w) = q(w) (1 - (1 - q(G))^N) / q(G)   w in G   (a match has a green candidate)
        q'(w) = q(w) (1 - q(G))^(N - 1)          w not in G   (all N candidates red)

    N = 2 is single-token non-distortionary (E_key q' = q); N > 2 is distortionary.
    Repeated context masking: a window already present earlier in the visible sequence is
    sampled without the watermark (the paper's 1-sequence masking; prompt windows count too).
    """

    def __init__(
        self,
        key: int,
        depth: int,
        vocab_size: int,
        context_width: int = 4,
        n_competitors: int = 2,
        temperature: float = 1.0,
        suppress_ids: Sequence[int] = (),
    ):
        super().__init__(temperature, suppress_ids)
        if not 1 <= depth <= 64:
            raise ValueError(f"depth must be in [1, 64] (one hash bit per layer), got {depth}")
        self.key, self.depth, self.vocab_size = key, depth, vocab_size
        self.context_width, self.n_competitors = context_width, n_competitors

    def g_values(self, context: Sequence[int], tokens: np.ndarray | None = None) -> np.ndarray:
        """[depth, len(tokens)] bool g-values for a context (default: the whole vocabulary)."""
        if tokens is None:
            tokens = np.arange(self.vocab_size)
        x = np.asarray(tokens, dtype=np.uint64) + np.uint64(1)
        h = _splitmix64(np.uint64(prf_seed(self.key, *context)) + x * _GOLDEN)
        bits = np.unpackbits(
            h.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1, bitorder="little"
        )
        return bits[:, : self.depth].T.astype(bool)

    def _watermark(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        h, n = self.context_width, self.n_competitors
        rows = [i for i, row in enumerate(input_ids) if not _context_repeats(row, h)]
        if not rows:
            return scores
        g = torch.from_numpy(np.stack([self.g_values(input_ids[i, -h:].tolist()) for i in rows]))
        q = torch.softmax(scores[rows].double(), dim=-1)
        for layer in range(self.depth):
            green = g[:, layer]
            mass = (q * green).sum(-1, keepdim=True).clamp(0.0, 1.0)  # rounding can exceed 1
            win = -torch.expm1(n * torch.log1p(-mass)) / mass.clamp_min(1e-300)
            q = q * torch.where(green, win, (1 - mass) ** (n - 1))
        out = scores.clone()
        out[rows] = torch.log(q).to(scores.dtype)
        return out


def kgw_z_score(ids: Sequence[int], start: int, wm: KGWWatermark) -> float:
    """Keyed KGW detector: green-token count z-score over positions >= start."""
    positions = range(max(start, 1), len(ids))
    n = len(positions)
    green = sum(bool(wm.green_mask(ids[t - 1])[ids[t]]) for t in positions)
    return (green - wm.gamma * n) / math.sqrt(n * wm.gamma * (1 - wm.gamma))


def gumbel_z_score(ids: Sequence[int], start: int, wm: GumbelWatermark) -> float:
    """Keyed Aaronson detector: sum of -log(1 - r_y) ~ Gamma(n, 1) under H0, as a z-score."""
    h = wm.context_width
    positions = range(max(start, h), len(ids))
    n = len(positions)
    score = sum(-math.log1p(-float(wm.uniforms(ids[t - h : t])[ids[t]])) for t in positions)
    return (score - n) / math.sqrt(n)


def tournament_z_score(ids: Sequence[int], start: int, wm: TournamentWatermark) -> float:
    """Keyed SynthID detector: layer-weighted mean g-value as a z-score over positions >= start.

    Positions whose h-token context occurred earlier in ids are skipped: masking left them
    unwatermarked and their g-values would repeat. The rest are iid Bernoulli(1/2) under H0.
    Weights fall linearly from 10 to 1 (the paper's weighted mean): deeper layers carry less
    evidence, because each layer spends some of the entropy the next one could have used.
    """
    h = wm.context_width
    weights = np.linspace(10, 1, wm.depth)
    seen: set[tuple[int, ...]] = set()
    total, n = 0.0, 0
    for t in range(h, len(ids)):
        ctx = tuple(ids[t - h : t])
        if ctx in seen:
            continue
        seen.add(ctx)
        if t >= start:
            total += float(weights @ (wm.g_values(ctx, np.array([ids[t]]))[:, 0] - 0.5))
            n += 1
    return total / math.sqrt(n * float(weights @ weights) / 4)
