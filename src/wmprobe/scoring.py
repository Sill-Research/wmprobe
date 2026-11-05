"""Teacher-forced proxy scoring, compressed into per-text sufficient statistics.

A full [T, V] distribution per text is too large to keep for a corpus, so each text is
reduced to: observed-token log-prob/entropy/variance at temperature 1 (Test A), the
top-k proxy logits (Test B), and logsumexp(s * logits) on a grid of inverse
temperatures s (temperature fitting, tempered probabilities, collision probabilities).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

# Inverse temperatures s = 1/tau. Test D needs lse(2s), so s_hat must stay <= 2.0.
S_GRID = np.round(np.arange(0.2, 4.0001, 0.1), 6)


@dataclass
class ScoredText:
    tokens: np.ndarray  # [T] observed tokens
    prev: np.ndarray  # [T] token preceding each observed token
    logit_y: np.ndarray  # [T] raw proxy logit of the observed token
    logp_y: np.ndarray  # [T] log p(y_t) at s=1
    entropy: np.ndarray  # [T] H(p_t) at s=1
    var_logp: np.ndarray  # [T] Var_{p_t}[log p_t] at s=1
    topk_idx: np.ndarray  # [T, k]
    topk_logit: np.ndarray  # [T, k]
    lse: np.ndarray  # [T, len(S_GRID)] logsumexp(s * logits)

    def __len__(self) -> int:
        return len(self.tokens)

    def lse_at(self, s: float) -> np.ndarray:
        return CubicSpline(S_GRID, self.lse, axis=1)(s)

    def topk_probs(self, s: float) -> np.ndarray:
        return np.exp(s * self.topk_logit - self.lse_at(s)[:, None])


@torch.no_grad()
def score_from_logits(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    prev: torch.Tensor,
    k: int = 100,
    exclude_ids: Sequence[int] = (),
) -> ScoredText:
    """logits [T, V] predicting tokens [T]; prev [T] is the token before each one.

    exclude_ids are scored as impossible (e.g. EOS: an interior token is never EOS).
    """
    logits = logits.float().clone()
    if exclude_ids:
        logits[:, list(exclude_ids)] = float("-inf")
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    plogp = torch.nan_to_num(p * logp)
    entropy = -plogp.sum(-1)
    var = torch.nan_to_num(plogp * logp).sum(-1) - entropy**2
    rows = torch.arange(len(tokens))
    top = torch.topk(logits, min(k, logits.shape[-1]), dim=-1)
    lse = torch.stack([torch.logsumexp(float(s) * logits, dim=-1) for s in S_GRID], dim=1)
    return ScoredText(
        tokens=tokens.numpy().astype(np.int64),
        prev=prev.numpy().astype(np.int64),
        logit_y=logits[rows, tokens].double().numpy(),
        logp_y=logp[rows, tokens].double().numpy(),
        entropy=entropy.double().numpy(),
        var_logp=var.clamp_min(0).double().numpy(),
        topk_idx=top.indices.numpy().astype(np.int32),
        topk_logit=top.values.numpy(),
        lse=lse.double().numpy(),
    )


@torch.no_grad()
def score_continuations(
    model,
    contexts: Sequence[Sequence[int]],
    continuations: Sequence[Sequence[int]],
    k: int = 100,
    batch_size: int = 16,
    exclude_ids: Sequence[int] = (),
) -> list[ScoredText]:
    """Score each continuation given its (visible) context with a causal LM."""
    out = []
    for i in range(0, len(contexts), batch_size):
        ctxs, conts = contexts[i : i + batch_size], continuations[i : i + batch_size]
        seqs = [list(c) + list(y) for c, y in zip(ctxs, conts, strict=True)]
        width = max(map(len, seqs))
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for j, seq in enumerate(seqs):
            ids[j, : len(seq)] = torch.tensor(seq)
            mask[j, : len(seq)] = 1
        logits = model(input_ids=ids, attention_mask=mask).logits
        for j, (c, y) in enumerate(zip(ctxs, conts, strict=True)):
            n_ctx, n_y = len(c), len(y)
            out.append(
                score_from_logits(
                    logits[j, n_ctx - 1 : n_ctx - 1 + n_y],
                    ids[j, n_ctx : n_ctx + n_y],
                    ids[j, n_ctx - 1 : n_ctx - 1 + n_y],
                    k=k,
                    exclude_ids=exclude_ids,
                )
            )
    return out
