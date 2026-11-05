"""A tiny deterministic language model for fast, exact detector tests.

Next-token logits depend on the previous two tokens, so the "proxy" (teacher-forced
logits) is available exactly, and repeated prompts reproduce identical distributions.
"""

import numpy as np
import torch

from wmprobe.scoring import ScoredText, score_from_logits

V = 200


class ToyLM:
    def __init__(self, vocab_size: int = V, seed: int = 0, scale: float = 2.0):
        g = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.a = torch.randn(vocab_size, vocab_size, generator=g) * scale
        self.b = torch.randn(vocab_size, vocab_size, generator=g) * (scale / 2)

    def next_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [B, L] (L >= 2) -> logits [B, V] for the next token."""
        return self.a[ids[:, -1]] + self.b[ids[:, -2]]

    def teacher_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [L] -> logits [L-2, V]; row i predicts ids[i + 2]."""
        return self.a[ids[1:-1]] + self.b[ids[:-2]]


def sample(lm: ToyLM, prompts: torch.Tensor, n_new: int, processor, seed: int = 0):
    """Autoregressively sample n_new tokens. prompts [B, L0] -> [B, L0 + n_new]."""
    g = torch.Generator().manual_seed(seed)
    ids = prompts.clone()
    for _ in range(n_new):
        scores = processor(ids, lm.next_logits(ids))
        probs = torch.softmax(scores, dim=-1)
        nxt = torch.multinomial(probs, 1, generator=g)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


def random_prompts(n: int, length: int = 4, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (n, length), generator=g)


def score(lm: ToyLM, ids: torch.Tensor, prompt_len: int, k: int = 50) -> ScoredText:
    """Teacher-force the continuation ids[prompt_len:] with the prompt visible."""
    logits = lm.teacher_logits(ids)[prompt_len - 2 :]
    tokens = ids[prompt_len:]
    prev = ids[prompt_len - 1 : -1]
    return score_from_logits(logits, tokens, prev, k=k)


def corpus(lm: ToyLM, processor, n_texts: int, n_new: int, seed: int) -> list[ScoredText]:
    prompts = random_prompts(n_texts, seed=seed)
    ids = sample(lm, prompts, n_new, processor, seed=seed + 1)
    return [score(lm, row, prompts.shape[1]) for row in ids]


def fraction(x: np.ndarray, cond) -> float:
    return float(np.mean(cond(np.asarray(x))))
