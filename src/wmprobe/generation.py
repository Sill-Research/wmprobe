"""GPT-2 generation through Hugging Face `generate`, with our processors as the sampler."""

from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList


def load(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    return model, tok


def _left_pad(prompts: Sequence[Sequence[int]], pad_id: int):
    width = max(map(len, prompts))
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros(len(prompts), width, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, width - len(p) :] = torch.tensor(p)
        mask[i, width - len(p) :] = 1
    return ids, mask


@torch.no_grad()
def generate(
    model,
    tok,
    prompts: Sequence[Sequence[int]],
    processor,
    max_new_tokens: int,
    seed: int,
    batch_size: int = 32,
) -> list[list[int]]:
    """Sample continuations (new token ids only). The processor owns temperature/watermark."""
    out = []
    for b, i in enumerate(range(0, len(prompts), batch_size)):
        ids, mask = _left_pad(prompts[i : i + batch_size], tok.eos_token_id)
        torch.manual_seed(seed * 100_003 + b)
        gen = model.generate(
            input_ids=ids,
            attention_mask=mask,
            logits_processor=LogitsProcessorList([processor]),
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        out += gen[:, ids.shape[1] :].tolist()
    return out


@torch.no_grad()
def next_token_probs(model, prompts: Sequence[Sequence[int]], processor) -> torch.Tensor:
    """Exact next-token sampling distribution [N, V] of a black box using `processor`."""
    rows = []
    for p in prompts:
        ids = torch.tensor([list(p)])
        logits = model(input_ids=ids).logits[:, -1]
        rows.append(torch.softmax(processor(ids, logits), dim=-1)[0])
    return torch.stack(rows)
