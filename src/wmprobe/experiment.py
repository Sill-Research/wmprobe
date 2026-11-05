"""Run every keyless test against plain and watermarked GPT-2 text; write results.json.

    uv run python -m wmprobe.experiment --out runs/full
    uv run python -m wmprobe.experiment --out runs/quick --quick

Generated corpora and proxy scores are cached under --out, so reruns resume.
"""

import argparse
import dataclasses
import json
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from wmprobe.detectors import (
    bigram_residual_test,
    context_interaction_test,
    diversity_collapse_test,
    fit_inverse_temperature,
    likelihood_deficit_z,
    select_answer_set,
)
from wmprobe.generation import generate, load, next_token_probs
from wmprobe.scoring import ScoredText, score_continuations
from wmprobe.stats import roc_auc
from wmprobe.watermarks import (
    GumbelWatermark,
    KGWWatermark,
    PlainSampler,
    TournamentWatermark,
    gumbel_z_score,
    kgw_z_score,
    tournament_z_score,
)

KEY = 15485863
GENERATOR = "gpt2"

OPENERS = [
    "The history of the city begins",
    "Scientists announced on Tuesday that",
    "My grandmother used to tell me",
    "The best way to learn a new language is",
    "In the early hours of the morning,",
    "The committee voted to approve",
    "Here is a simple recipe for",
    "The economy grew faster than expected",
    "When I first moved to the countryside,",
    "The new smartphone features",
    "According to local police,",
    "The football season opened with",
    "Climate researchers have long warned that",
    "She opened the letter and",
    "The museum's latest exhibition",
    "To install the software, first",
    "The ancient Romans built",
    "Our quarterly results show",
    "The village was quiet until",
    "Doctors recommend that patients",
    "The film follows a young detective who",
    "In a surprising turn of events,",
    "The river flows through",
    "Many people believe that",
    "The startup raised funding to",
    "Last summer, we travelled to",
    "The theory of evolution explains",
    "The mayor told reporters that",
    "A good night's sleep is important because",
    "The band released their third album",
    "The spacecraft entered orbit around",
    "Students at the university protested",
    "The old lighthouse stood",
    "The recipe calls for two cups of",
    "Historians disagree about why",
    "The company's chief executive said",
    "On the first day of school,",
    "The forest is home to",
    "Experts say the best time to visit",
    "The trial began with",
]

# Test C: few-shot lists constrain a base LM's next token to a small answer space. Variants
# of a list differ only in the separator, which is also the token right before the answer.
LISTS = [
    ["apple", "banana", "cherry", "grape"],
    ["red", "blue", "green", "yellow"],
    ["dog", "cat", "horse", "rabbit"],
    ["France", "Germany", "Spain", "Italy"],
    ["one", "two", "three", "four"],
    ["carrot", "potato", "onion", "spinach"],
    ["London", "Paris", "Berlin", "Madrid"],
    ["piano", "guitar", "violin", "drums"],
    ["football", "tennis", "golf", "rugby"],
    ["Mercury", "Venus", "Earth", "Mars"],
]
SEPARATORS = [",", ";", " |", " /", " -", ":", " +", " &", " ~", " *", " >"]


def list_prompt(items: list[str], sep: str) -> str:
    return sep.join([items[0]] + [" " + i for i in items[1:]]) + sep


@dataclass
class Config:
    out: Path
    n_prompts: int = 200
    n_new: int = 232
    burn_in: int = 32  # Tests A/B skip the first tokens, where the hidden prompt matters most
    lengths: tuple[int, ...] = (25, 50, 100, 200)
    corpus_sizes: tuple[int, ...] = (5, 10, 25, 50, 100)
    n_draws: int = 20
    c_queries: tuple[int, ...] = (50, 200, 1000)
    c_reps: int = 10
    c_variants: int = 8
    d_prompts: int = 40
    d_repeats: int = 10
    d_new: int = 32
    proxies: tuple[str, ...] = ("gpt2", "distilgpt2")
    conditions: tuple[str, ...] | None = None  # None = all
    seed: int = 0
    log: Callable[[str], None] = field(default=print, repr=False)


QUICK = dict(
    n_prompts=40,
    n_new=132,
    lengths=(25, 50, 100),
    corpus_sizes=(5, 10, 20),
    n_draws=10,
    c_queries=(50, 200),
    c_reps=4,
    d_prompts=10,
)


def make_conditions(vocab_size: int, eos: int) -> dict:
    sup = [eos]
    conds: dict = {f"plain_t{t}": PlainSampler(t, sup) for t in (0.7, 1.0, 1.2)}
    for g in (0.25, 0.5):
        for d in (1, 2, 4):
            conds[f"kgw_g{g}_d{d}"] = KGWWatermark(KEY, g, float(d), vocab_size, 1.0, sup)
    for h in (4, 1):
        conds[f"gumbel_h{h}"] = GumbelWatermark(KEY, h, vocab_size, 1.0, sup)
    # SynthID-Text paper defaults: m=30 layers, H=4 context; N=2 is non-distortionary.
    for n in (2, 4):
        conds[f"synthid_n{n}"] = TournamentWatermark(KEY, 30, vocab_size, 4, n, 1.0, sup)
    return conds


def is_plain(name: str) -> bool:
    return name.startswith("plain")


def _cached(path: Path, compute: Callable, as_json: bool = False):
    if path.exists():
        return json.loads(path.read_text()) if as_json else pickle.loads(path.read_bytes())
    value = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_json:
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(pickle.dumps(value))
    return value


def auc_table(scores: dict) -> dict:
    """{wm: {plain: AUC}} for every watermarked/plain pair; scores[cond] -> list or dict."""
    out = {}
    for wm in (c for c in scores if not is_plain(c)):
        out[wm] = {}
        for pl in (c for c in scores if is_plain(c)):
            pos, neg = scores[wm], scores[pl]
            if isinstance(pos, dict):
                out[wm][pl] = {k: _safe_auc(pos[k], neg[k]) for k in pos}
            else:
                out[wm][pl] = _safe_auc(pos, neg)
    return out


def _safe_auc(pos, neg) -> float | None:
    pos = [p for p in pos if p is not None and np.isfinite(p)]
    neg = [n for n in neg if n is not None and np.isfinite(n)]
    return roc_auc(pos, neg) if pos and neg else None


# ---------------------------------------------------------------- data


def build_prompts(model, tok, cfg: Config) -> list[list[int]]:
    per_opener = -(-cfg.n_prompts // len(OPENERS))
    base = [tok.encode(o) for o in OPENERS for _ in range(per_opener)][: cfg.n_prompts]
    ext = generate(model, tok, base, PlainSampler(1.0, [tok.eos_token_id]), 12, seed=10_000)
    return [b + e for b, e in zip(base, ext, strict=True)]


def build_corpus(model, tok, prompts, proc, n_new: int, seed: int) -> list[dict]:
    new = generate(model, tok, prompts, proc, n_new, seed=seed)
    return [
        {"prompt": p, "new": n, "text": tok.decode(n)} for p, n in zip(prompts, new, strict=True)
    ]


def score_hidden_prompt(proxy, tok, corpus: list[dict]) -> list[ScoredText]:
    """What a keyless detector sees: only the text, re-tokenized, after <|endoftext|>.

    Interior tokens are never EOS, so the proxy is conditioned on "not EOS".
    """
    conts = [tok.encode(item["text"]) for item in corpus]
    eos = tok.eos_token_id
    return score_continuations(proxy, [[eos]] * len(conts), conts, exclude_ids=(eos,))


def skip(text: ScoredText, n: int) -> ScoredText:
    return ScoredText(**{f.name: getattr(text, f.name)[n:] for f in dataclasses.fields(text)})


# ---------------------------------------------------------------- analyses


def keyed_baselines(tok, corpora: dict, conds: dict) -> dict:
    out = {}
    for name, proc in conds.items():
        if is_plain(name):
            continue
        if isinstance(proc, KGWWatermark):
            detect = lambda ids, p=proc: kgw_z_score(ids, 1, p)  # noqa: E731
        elif isinstance(proc, TournamentWatermark):
            detect = lambda ids, p=proc: tournament_z_score(ids, p.context_width, p)  # noqa: E731
        else:
            detect = lambda ids, p=proc: gumbel_z_score(ids, p.context_width, p)  # noqa: E731
        z = [detect(tok.encode(it["text"])) for it in corpora[name]]
        entry = {"z": z}
        if "plain_t1.0" in corpora:
            neg = [detect(tok.encode(it["text"])) for it in corpora["plain_t1.0"]]
            entry["auc_vs_plain_t1.0"] = roc_auc(z, neg)
        out[name] = entry
    return out


def analyse_a(scored: dict, cfg: Config) -> dict:
    per, by_len = {}, {}
    for c, texts in scored.items():
        z = [likelihood_deficit_z(t) for t in texts]
        per[c] = {"z": z, "mean_z": float(np.mean(z))}
        # Watermarks push Z negative, so -Z is the detection score.
        by_len[c] = {
            str(L): [-likelihood_deficit_z(t, max_len=L) for t in texts] for L in cfg.lengths
        }
    return {"per_condition": per, "auc": auc_table(by_len)}


def analyse_b(scored: dict, raw: dict, cfg: Config) -> dict:
    """scored: texts after burn-in; raw: the same texts without burn-in (for comparison)."""
    rng = np.random.default_rng(cfg.seed)
    per, draws_all = {}, {}
    for c, texts in scored.items():
        full = bigram_residual_test(texts)
        z, p = {}, {}
        for m in cfg.corpus_sizes:
            if m > len(texts):
                continue
            res = [
                bigram_residual_test([texts[i] for i in rng.choice(len(texts), m, replace=False)])
                for _ in range(cfg.n_draws)
            ]
            z[str(m)], p[str(m)] = [r.z for r in res], [r.p_value for r in res]
        per[c] = {
            "full_z": full.z,
            "full_p": full.p_value,
            "full_p_no_burn_in": bigram_residual_test(raw[c]).p_value,
            "s_hat": full.s,
            "draws_z": z,
            "draws_p": p,
            "reject05": {m: float(np.mean(np.array(v) < 0.05)) for m, v in p.items()},
        }
        draws_all[c] = z
    return {"per_condition": per, "auc": auc_table(draws_all)}


def analyse_c(gen_model, tok, proxies: dict, conds: dict, cfg: Config) -> dict:
    seps = [s for s in SEPARATORS if len(tok.encode(s)) == 1][: cfg.c_variants]
    templates = [[tok.encode(list_prompt(items, s)) for s in seps] for items in LISTS]
    with torch.no_grad():
        proxy_logits = {
            name: [
                torch.stack([m(input_ids=torch.tensor([p])).logits[0, -1] for p in tpl]).numpy()
                for tpl in templates
            ]
            for name, m in proxies.items()
        }
    for logits in proxy_logits.values():  # the black box never answers with EOS
        for lg in logits:
            lg[:, tok.eos_token_id] = -np.inf
    gen = torch.Generator().manual_seed(cfg.seed)
    raw = {name: {} for name in proxies}
    for c, proc in conds.items():
        q = [next_token_probs(gen_model, tpl, proc) for tpl in templates]
        for name in proxies:
            raw[name][c] = {str(n): {"z": [], "m": []} for n in cfg.c_queries}
        for n in cfg.c_queries:
            for _ in range(cfg.c_reps):
                for ti, qt in enumerate(q):
                    draws = torch.multinomial(qt, n, replacement=True, generator=gen)
                    counts = torch.stack(
                        [torch.bincount(row, minlength=qt.shape[1]) for row in draws]
                    ).numpy()
                    for name in proxies:
                        logits = proxy_logits[name][ti]
                        w = select_answer_set(logits, n)
                        slot = raw[name][c][str(n)]
                        slot["m"].append(len(w))
                        try:
                            res = context_interaction_test(counts[:, w], logits[:, w])
                            slot["z"].append(res.z)
                        except ValueError:  # < 2 answers, or < 2 variants answered inside w
                            slot["z"].append(None)
    out = {}
    for name in proxies:
        per = {
            c: {
                n: {
                    "z": v["z"],
                    "reject01": float(np.mean([z is not None and z > 2.326 for z in v["z"]])),
                    "answer_set_size": float(np.mean(v["m"])),
                }
                for n, v in by_n.items()
            }
            for c, by_n in raw[name].items()
        }
        scores = {c: {n: v["z"] for n, v in by_n.items()} for c, by_n in raw[name].items()}
        out[name] = {"per_condition": per, "auc": auc_table(scores), "separators": seps}
    return out


def generate_repeats(model, tok, prompts, proc, cfg: Config, seed: int) -> list[list[int]]:
    batch = [p for p in prompts for _ in range(cfg.d_repeats)]
    return generate(model, tok, batch, proc, cfg.d_new, seed=seed)


def analyse_d(tok, proxy, prompts, samples: dict, cfg: Config) -> dict:
    per, scores = {}, {}
    r = cfg.d_repeats
    contexts = [p for p in prompts for _ in range(r)]
    for c, new in samples.items():
        conts = [tok.encode(tok.decode(n)) for n in new]
        scored = score_continuations(proxy, contexts, conts, exclude_ids=(tok.eos_token_id,))
        s = fit_inverse_temperature(scored)
        groups = [scored[i * r : (i + 1) * r] for i in range(len(prompts))]
        pairs = [list(zip(g[0::2], g[1::2], strict=True)) for g in groups]
        try:
            z = [diversity_collapse_test(p, s).z for p in pairs]
            pooled = diversity_collapse_test([x for p in pairs for x in p], s).z
        except ValueError:  # fitted temperature outside the scoring grid
            z, pooled = [None] * len(pairs), None
        per[c] = {
            "z": z,
            "pooled_z": pooled,
            "s_hat": s,
            "reject05": float(np.mean([v is not None and v > 1.645 for v in z])),
        }
        scores[c] = z
    return {"per_condition": per, "auc": auc_table(scores)}


# ---------------------------------------------------------------- driver


def run(cfg: Config) -> dict:
    t0 = time.time()

    def log(msg: str) -> None:
        cfg.log(f"[{time.time() - t0:7.1f}s] {msg}")

    torch.manual_seed(cfg.seed)
    out = Path(cfg.out)
    gen_model, tok = load(GENERATOR)
    conds = make_conditions(gen_model.config.vocab_size, tok.eos_token_id)
    if cfg.conditions is not None:
        conds = {c: conds[c] for c in cfg.conditions}

    prompts = _cached(out / "prompts.json", lambda: build_prompts(gen_model, tok, cfg), True)
    corpora = {}
    for i, (c, proc) in enumerate(conds.items()):
        log(f"corpus {c}")
        corpora[c] = _cached(
            out / "corpus" / f"{c}.json",
            lambda proc=proc, i=i: build_corpus(gen_model, tok, prompts, proc, cfg.n_new, i),
            True,
        )
    d_prompts = prompts[: cfg.d_prompts]
    repeats = {}
    for i, (c, proc) in enumerate(conds.items()):
        log(f"repeats {c}")
        repeats[c] = _cached(
            out / "repeats" / f"{c}.json",
            lambda proc=proc, i=i: generate_repeats(gen_model, tok, d_prompts, proc, cfg, 500 + i),
            True,
        )

    results: dict = {
        "config": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "log"},
        "keyed": keyed_baselines(tok, corpora, conds),
        "test_a": {},
        "test_b": {},
        "test_d": {},
    }
    results["config"]["out"] = str(cfg.out)
    proxies = {name: (gen_model if name == GENERATOR else load(name)[0]) for name in cfg.proxies}
    for name, proxy in proxies.items():
        scored = {}
        for c in conds:
            log(f"score {name} / {c}")
            scored[c] = _cached(
                out / "scored" / name / f"{c}.pkl",
                lambda c=c, proxy=proxy: score_hidden_prompt(proxy, tok, corpora[c]),
            )
        log(f"test A/B with {name}")
        burned = {c: [skip(t, cfg.burn_in) for t in texts] for c, texts in scored.items()}
        results["test_a"][name] = analyse_a(burned, cfg)
        results["test_b"][name] = analyse_b(burned, scored, cfg)
        del scored
        log(f"test D with {name}")
        results["test_d"][name] = analyse_d(tok, proxy, d_prompts, repeats, cfg)
    log("test C")
    results["test_c"] = analyse_c(gen_model, tok, proxies, conds, cfg)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    log(f"wrote {out / 'results.json'}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="small configuration (~10 min on CPU)")
    args = ap.parse_args()
    cfg = Config(out=args.out, **(QUICK if args.quick else {}))
    run(cfg)


if __name__ == "__main__":
    main()
