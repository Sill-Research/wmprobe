# wmprobe — keyless LLM watermark detection experiments

Research code. **Development has stopped**; this README records what was tried and what was found.

Question: can a third party tell that text (or a model) is watermarked **without the secret key**?
Three watermark schemes were implemented on GPT-2, and four keyless tests were run against each.
The starting proposal is [`watermark-detection.md`](watermark-detection.md); the SynthID-Text
reference is `SynthID-Text.pdf` (Dathathri et al., *Nature* 634, 2024).

- Generator: GPT-2.
- Proxies (the model a keyless detector re-scores text with): GPT-2 itself (same proxy, best
  case) and DistilGPT-2 (mismatched proxy, realistic case). Both share GPT-2's tokenizer.

```
uv sync
uv run pytest -m "not slow"                    # 63 fast tests on an exact toy LM (~15 s)
uv run pytest                                  # all 77, incl. GPT-2 integration + pipeline smoke test
uv run python -m wmprobe.experiment --out runs/full      # full sweep, ~1.5 h on CPU; resumes from cache
uv run python -m wmprobe.experiment --out runs/quick --quick
uv run python -m wmprobe.report runs/full      # -> runs/full/report.md + figures/
uv run ruff check src tests
```

| module | contents |
|---|---|
| `watermarks.py` | `PlainSampler`, `KGWWatermark`, `GumbelWatermark`, `TournamentWatermark` (HF `LogitsProcessor`s); keyed detectors |
| `scoring.py` | teacher-forced proxy scoring → per-text sufficient statistics (`ScoredText`) |
| `detectors.py` | keyless Tests A–D, temperature MLE |
| `generation.py` | batched GPT-2 sampling through `generate`; exact next-token distributions |
| `experiment.py` / `report.py` | full sweep → `results.json` → `report.md` + figures |

## Notation

```
p_t(w)     model's next-token distribution at step t (after temperature)
q_t(w)     watermarked sampling distribution at step t
k          secret key;  r_t = PRF(k, recent tokens)  per-step seed
y_1..y_T   observed tokens
H0: y_t ~ p_t (not watermarked)        H1: y_t ~ q_t^(k) for an unknown key k
q̄_t(w) = E_k[ q_t^(k)(w) ]             what a keyless observer sees, averaged over keys
```

If `q̄_t = p_t` exactly, no single-text keyless test can work; any signal must come from
structure across samples.

## Watermarking schemes implemented

Every processor returns logits whose softmax is the exact sampling distribution. Temperature is
applied first, so the watermark acts on the tempered distribution. EOS is suppressed.

### Plain sampling (control)

```
q_t(w) = p_t(w) ∝ exp(ℓ_t(w) / τ)          τ ∈ {0.7, 1.0, 1.2}
```

### KGW red-green (Kirchenbauer et al. 2023)

Green list `G(y_{t−1}, k)`: a pseudorandom `γ` fraction of the vocabulary, seeded by the
previous token and the key.

```
q_t(w) ∝ p_t(w) · exp(δ · 1[w ∈ G(y_{t−1}, k)])           γ ∈ {0.25, 0.5},  δ ∈ {1, 2, 4}

keyed detector:  z = (|s|_G − γT) / sqrt(T γ (1−γ))       |s|_G = number of green tokens
```

Distortionary: boosting a random set flattens the distribution on average, so `q̄_t ≠ p_t`.

### Gumbel / Aaronson (distortion-free sampling)

```
r_{t,w} = PRF(k, y_{t−h..t−1}, w) ~ Uniform(0,1)           h ∈ {1, 4}
y_t = argmax_w  r_{t,w}^(1 / p_t(w))

keyed detector:  S = Σ_t −log(1 − r_{t,y_t}) ~ Gamma(n, 1) under H0,   z = (S − n) / sqrt(n)
```

This is the Gumbel-max trick, so `P_k(y_t = w) = p_t(w)` and `q̄_t = p_t`. For a fixed key,
generation is deterministic given the last `h` tokens.

### SynthID-Text — Tournament sampling (Dathathri et al. 2024)

Settings follow the paper's defaults: `m = 30` layers, Bernoulli(½) g-values, a seed from the
last `H = 4` tokens, and 1-sequence repeated-context masking.

```
seed       r_t = PRF(k, y_{t−4..t−1})
g-values   g_ℓ(w, r_t) ∈ {0, 1},  ℓ = 1..m       (bit ℓ of splitmix64(r_t, w); iid Bernoulli(½))
```

**Sampling (Algorithm 2).** Draw `N^m` candidates from `p_t`, then run `m` knockout layers. In
layer `ℓ`, groups of `N` compete, the highest `g_ℓ` wins and ties are broken uniformly; the last
winner is `y_t`. `N = 2` is the non-distortionary mode and `N > 2` is distortionary. Tried:
`N ∈ {2, 4}`.

**Closed form (used by the implementation).** Let `q⁽⁰⁾ = p_t`. For layer `ℓ`, let
`G = {w : g_ℓ(w) = 1}` and `P = q⁽ℓ⁻¹⁾(G)`. A match is won by a green candidate unless all
`N` candidates are red:

```
q⁽ℓ⁾(w) = q⁽ℓ⁻¹⁾(w) · (1 − (1 − P)^N) / P        w ∈ G
q⁽ℓ⁾(w) = q⁽ℓ⁻¹⁾(w) · (1 − P)^(N−1)              w ∉ G
q_t = q⁽ᵐ⁾
```

For `N = 2` this reduces to `q⁽ℓ⁾(w) = q⁽ℓ⁻¹⁾(w) · (1 + g_ℓ(w) − P)`, and
`E_k[q⁽ℓ⁾] = q⁽ℓ⁻¹⁾` (single-token non-distortion). A test checks the closed form against a
literal Monte Carlo run of Algorithm 2 for `N = 2` and `N = 3`.

**Repeated-context masking.** If the window `y_{t−4..t−1}` already occurred earlier in the
sequence, `y_t ~ p_t` (no watermark). Here prompt windows count too, a small departure from the
paper, which counts only windows used during generation.

**Keyed detector.** Score only positions whose 4-token context is new within the text; their
g-values are iid Bernoulli(½) under H0. Layer weights `w_ℓ` fall linearly from 10 to 1:

```
z = Σ_t Σ_ℓ w_ℓ (g_ℓ(y_t, r_t) − ½) / sqrt(n · Σ_ℓ w_ℓ² / 4)          n = scored positions
```

**Properties used below:**

```
N = 2, single text:  E_k[q_t] = p_t at every step, and masking gives each scored step a fresh
                     seed  ⇒  the whole text has exactly the H0 distribution (no keyless
                     single-text test can work)
N > 2:               q̄_t ≠ p_t; the key-average is flatter (the high-p token loses whenever
                     it is outnumbered by green candidates)
fixed key:           Σ_w q_t(w)² ≫ Σ_w p_t(w)²   (measured on GPT-2: 0.64 vs 0.22,
                     median ratio 5×)  ⇒  repeated samples of one prompt agree too often
```

## Keyless tests

All tests use proxy teacher forcing: one forward pass per text gives `p_t` at every position.
The detector sees only the text, re-tokenized after `<|endoftext|>`. EOS is excluded from the
proxy distribution, since interior tokens are never EOS. Tests A and B skip the first 32
tokens, where the hidden prompt matters most.

**Temperature MLE** (used by Tests B and D):

```
ŝ = argmax_s Σ_t [ s·ℓ_t(y_t) − log Σ_w exp(s·ℓ_t(w)) ]          ŝ = 1/τ̂, proxy logits ℓ_t
```

### Test A — single text, likelihood deficit

```
S_t = log p_t(y_t) + H(p_t)              E_H0[S_t] = 0,   Var_H0[S_t] = Var_{p_t}[log p_t]
Z   = Σ_t S_t / sqrt( Σ_t Var_H0[S_t] )  ≈ N(0,1) under H0 (martingale CLT); negative = watermark-like
```

### Test B — corpus, cross-text bigram residuals

```
R_j(a,b) = Σ_{t: y_{t−1}=a} ( 1[y_t = b] − p_t^ŝ(b) ) · 1[b ∈ top-k_t]      per text j
T        = Σ_{j<j'} ⟨R_j, R_j'⟩                                            E_H0[T] = 0
```

The p-value comes from randomly flipping the sign of each `R_j`.

Changes from the proposal:
- Temperature is fitted out first, because temperature alone makes residual signs consistent
  per `(a, b)`.
- Cross-text products keep the statistic zero-mean.
- The sign-flip null is used because one cell (`"." → "\n"`) can dominate, making `T`
  χ²₁-like.

### Test C — query access, context × answer interaction

Few-shot list prompts (`apple, banana, cherry, grape,`) whose `K` variants differ only in the
separator, which is the token before the answer. Each variant is queried repeatedly. Answers are
restricted to a set with expected count ≥ 5 in every variant.

```
H0:  π_c(w) ∝ exp( s·ℓ_c(w) + β_w )          temperature s, per-word proxy error β_w
G² = 2 Σ_{c,w} n_{c,w} log( n_{c,w} / n̂_{c,w} )  vs  χ²_{K(M−1)−M}
```

Change from the proposal: a χ² independence test is invalid, because the preceding token
legitimately changes the answer distribution under H0.

### Test D — query access, diversity collapse

Repeat each prompt (10 samples × 32 tokens) and pair samples disjointly. Walk each pair while
the prefixes are identical, stopping at the first disagreement.

```
c_t = Σ_w p_t^ŝ(w)²                         H0 probability that the next tokens agree
z   = (O − Σ_t c_t) / sqrt( Σ_t c_t (1 − c_t) )      O = observed agreements
```

`ŝ` is fitted from the samples' own likelihoods, which separates "low temperature" from
"concentrated at normal temperature".

## Results

Full run: 200 texts × 232 tokens per condition (200 analysed after burn-in). Test C uses 10
lists × 8 separator variants. Test D uses 40 prompts × 10 samples. All tables and figures are in
`runs/full/report.md`.

**Keyed baselines** (AUC vs plain T=1.0, 200 texts each):

| scheme | KGW γ.25 δ1/2/4 | KGW γ.5 δ1/2/4 | Gumbel h4 / h1 | SynthID N=2 / N=4 |
|---|---|---|---|---|
| mean z | 5.9 / 13.0 / 22.1 | 6.1 / 10.3 / 13.8 | 51.4 / 53.3 | 23.9 / 42.4 |
| AUC | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

**Keyless tests.** Columns:
- **A:** AUC at 200 tokens vs plain T=1.0 / vs plain T=1.2.
- **B:** rejection rate at p<0.05, with corpora of 25 / 100 texts.
- **C:** rejection rate at p<0.01, with 1000 queries per variant.
- **D:** pooled z over 40 prompts / per-prompt rejection rate at p<0.05.

Plain rows are false-positive rates.

Same proxy (GPT-2):

| condition | A | B | C | D |
|---|---|---|---|---|
| plain T=0.7 | — | 0.00 / 0.00 | 0.00 | 0.1 / 0.10 |
| plain T=1.0 | — | 0.05 / 0.05 | 0.02 | 0.9 / 0.03 |
| plain T=1.2 | — | 0.05 / 0.00 | 0.01 | 0.4 / 0.07 |
| KGW γ.25 δ1 | 0.55 / 0.00 | 1.00 / 1.00 | 1.00 | −1.6 / 0.05 |
| KGW γ.25 δ2 | 0.78 / 0.00 | 1.00 / 1.00 | 1.00 | 1.5 / 0.15 |
| KGW γ.25 δ4 | 0.96 / 0.00 | 1.00 / 1.00 | 1.00 | 0.9 / 0.17 |
| KGW γ.5 δ1 | 0.57 / 0.00 | 0.95 / 1.00 | 1.00 | −0.3 / 0.03 |
| KGW γ.5 δ2 | 0.67 / 0.00 | 1.00 / 1.00 | 1.00 | 1.5 / 0.20 |
| KGW γ.5 δ4 | 0.82 / 0.00 | 1.00 / 1.00 | 1.00 | 3.0 / 0.28 |
| Gumbel h4 | 0.37 / 0.00 | 0.10 / 0.15 | 0.30 | 200.4 / 1.00 |
| Gumbel h1 | 0.03 / 0.00 | 1.00 / 1.00 | 0.50 | 156.6 / 1.00 |
| SynthID N=2 | 0.48 / 0.00 | 0.05 / 0.10 | 1.00 | 39.8 / 0.85 |
| SynthID N=4 | 1.00 / 0.96 | 0.10 / 0.50 | 0.36 | 126.4 / 0.97 |

Mismatched proxy (DistilGPT-2):

| condition | A | B | C | D |
|---|---|---|---|---|
| plain T=0.7 | — | 0.90 / 1.00 | 1.00 | 2.0 / 0.15 |
| plain T=1.0 | — | 0.90 / 1.00 | 1.00 | 1.3 / 0.07 |
| plain T=1.2 | — | 0.20 / 1.00 | 1.00 | 0.5 / 0.12 |
| KGW γ.25 δ1 | 0.52 / 0.00 | 1.00 / 1.00 | 1.00 | −1.0 / 0.07 |
| KGW γ.25 δ2 | 0.73 / 0.00 | 1.00 / 1.00 | 1.00 | 2.1 / 0.12 |
| KGW γ.25 δ4 | 0.93 / 0.01 | 1.00 / 1.00 | 1.00 | 1.2 / 0.20 |
| KGW γ.5 δ1 | 0.55 / 0.00 | 1.00 / 1.00 | 1.00 | 0.2 / 0.07 |
| KGW γ.5 δ2 | 0.65 / 0.00 | 1.00 / 1.00 | 1.00 | 1.8 / 0.23 |
| KGW γ.5 δ4 | 0.78 / 0.00 | 1.00 / 1.00 | 1.00 | 4.2 / 0.25 |
| Gumbel h4 | 0.38 / 0.01 | 0.90 / 1.00 | 0.50 | 219.0 / 1.00 |
| Gumbel h1 | 0.04 / 0.00 | 1.00 / 1.00 | 0.60 | 170.9 / 1.00 |
| SynthID N=2 | 0.50 / 0.00 | 0.60 / 1.00 | 1.00 | 41.9 / 0.85 |
| SynthID N=4 | 1.00 / 0.86 | 0.20 / 0.90 | 0.40 | 140.2 / 0.97 |

**Test D query budget for SynthID N=2.** This was computed by one-off scripts, not
`experiment.py`. The decision threshold is z > 3.09 (nominal p < 0.001); the plain rate is the
worst over T ∈ {0.7, 1.0, 1.2}.

| queries | layout | SynthID detected (GPT-2 / DistilGPT-2) | plain flagged (GPT-2 / DistilGPT-2) |
|---|---|---|---|
| 10 | 1 prompt × 10 | 73% / 68% | 0% / 4% |
| 20 | 2 prompts × 10 | 94% / 94% | 0% / 4% |
| 50 | 5 prompts × 10 | 100% / 99% | 0% / 6% |
| 100 | 10 prompts × 10 | 100% / 100% | 0% / 4% |
| 50 | 1 prompt × 50 (20 prompts tested) | 95% / 95% | 0% / 15% |

With a single prompt and the mismatched proxy, the false positives come from the same 2 of 20
prompts at every temperature. They are systematic proxy error, and more samples of that prompt
do not remove them.

## Findings

| access | KGW | Gumbel | SynthID N=2 | SynthID N=4 |
|---|---|---|---|---|
| Keyed | detected | detected | detected | detected |
| One text + proxy (A) | weak; confounded with temperature | no | no (provably) | yes; confounded with temperature |
| Corpus + proxy (B) | yes, same proxy only | h1 only, same proxy only | no | weak |
| Queries + proxy (C) | yes, same proxy only | partial | yes, same proxy only | partial |
| Repeated queries (D) | no | yes, both proxies | yes, both proxies (~50 queries) | yes, both proxies |

- **Proxy mismatch.** With DistilGPT-2, Tests B and C flag plain text at the same rate as
  watermarked text: they detect the proxy mismatch, not the watermark. Under Test A, plain
  T=1.0 has a mean Z of −3.0 instead of 0.
- **Test D is the only test that survives a mismatched proxy.** It detects schemes that are
  deterministic or concentrated for a fixed key: Gumbel and SynthID.
- **SynthID N=2.** Its samples are individually exact T=1 samples (fitted τ = 1.00). Only
  1.3% of positions in the plain corpus have a 4-token context shared with another text, so
  Test B has nothing to work with.
- **SynthID N=4 at m=30** produces near-gibberish. It is a stress setting, not the paper's
  quality-tuned distortionary configuration.
- **Gumbel h1** degenerates into repetition (82% repeated 4-grams vs 0.7% for plain text).
  This is why its Test A Z is positive.
- **What Test D establishes.** It tests whether a model watermarks its sampling. It does not
  test whether a given text found elsewhere is watermarked.

## Current state

- **Stopped.** All 77 tests pass and ruff is clean. `runs/` (gitignored) holds the cached
  full run and its `results.json` / `report.md`.
- **Not in the repo.** The query-budget numbers above came from one-off scripts.
- **Not implemented:**
  - cryptographically undetectable schemes (Christ et al.)
  - cache-augmented variants
  - paraphrase attacks
  - top-k / top-p truncation in generation
  - two-sample tests against an unwatermarked reference
  - per-text verification by querying a model at a text's own contexts
