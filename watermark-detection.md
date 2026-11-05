# Detecting LLM Watermarks Without the Key — Findings

Goal: determine whether a piece of LLM-generated text carries a watermark **without access to the secret key**, without needing to identify the exact scheme or key.

---

## 1. Background: token-level analysis of LLM text

- Any LLM text can be re-scored by a model (the same one or an open-weights proxy) via **teacher forcing**: one forward pass gives, for every position, the probability and rank of the actual token plus the full distribution over alternatives.
- LLM text is overwhelmingly composed of high-rank (top-10) tokens; human text is "speckled" with low-rank choices. This is the basis of GLTR (2019), DetectGPT / Fast-DetectGPT, and Binoculars.
- From output alone you can approximate the per-token distributions (blurred by the hidden prompt, unknown temperature/top-p, and proxy-model mismatch). You **cannot** recover model weights — a paragraph carries far too little information.

---

## 2. Formal setup

Let `p_t(·)` be the model's next-token distribution at step `t`. A watermark replaces it with a key-dependent `q_t^(k)(·)`. Observing tokens `y_1..y_T`, test:

- **H0:** `y_t ~ p_t` (no watermark)
- **H1:** `y_t ~ q_t^(k)` for some unknown key `k`

Without the key, what's observable is the key-averaged distribution:

```
q̄_t(w) = E_k[ q_t^(k)(w) ]
```

**Central fact:** if `q̄_t = p_t` exactly, no single-text test can work — watermarked and unwatermarked text are identically distributed. Any remaining signal must come from structure *across* samples (repeated contexts, corpus statistics, repeated queries).

---

## 3. Family 1 — Red-Green (Kirchenbauer et al., "KGW")

**Mechanism.** Hash the previous token with the key → split the vocab into a green fraction `γ` and a red remainder. Add `δ` to green logits:

```
q_t(w) ∝ p_t(w) · exp(δ · 1[w ∈ G(y_{t-1}, k)])
```

**Keyed detection.** Count green tokens `|s|_G`:

```
z = (|s|_G − γT) / sqrt(T·γ·(1−γ))
```

**Keyless detection — the scheme is distribution-modifying.** Boosting a random half of the vocab flattens the distribution on average (a top token at p=0.9 loses a lot when red, gains little when green). So `q̄_t ≠ p_t`.

### Test A — single text + proxy model (likelihood deficit)

Per-token centered log-likelihood:

```
S_t = log p_t(y_t) + H(p_t)
```

- Under H0: `E[S_t] = 0`, `Var(S_t) = Var_{p_t}[log p_t]` (both computable from the proxy).
- Under H1: `E[S_t] = Σ_w (q̄_t(w) − p_t(w)) · log p_t(w) < 0` (chosen tokens are slightly less likely than expected).

Aggregate:

```
Z = Σ_t S_t / sqrt( Σ_t Var(S_t) )
```

Strongly negative `Z` → evidence of watermark.

**Major confound:** higher sampling temperature (or an unusual hidden prompt) produces the *same* signature. From one text you cannot separate "watermark" from "temperature 1.2".

### Test B — corpus bigram residuals (breaks the confound)

Temperature flattens every distribution uniformly; KGW biases a *fixed* token set for each preceding token `a`. Over a corpus:

```
r(a, b) = count(a → b) − Σ_{t : y_{t−1} = a} p_t(b)
```

Under watermark, for each `a` the residuals split into a positive (green) and negative (red) group with **consistent sign across unrelated contexts**. Detect via permutation test or low-rank structure in the residual matrix. (This is essentially the "watermark stealing" approach of Jovanović et al., 2024.)

### Test C — query access (context-token dependence)

Use prompts with a constrained answer space ("name a random fruit"). Vary only the token immediately preceding the answer. Under H0 the answer distribution shouldn't depend on it systematically; under KGW it shifts. Use a χ² test of independence.

---

## 4. Family 2 — Distortion-free sampling (Aaronson/Gumbel, Kuditipudi et al.)

**Mechanism.** Pseudorandom `r_{t,w} = PRF_k(context, w)`, then:

```
y_t = argmax_w  r_{t,w}^(1 / p_t(w))
```

This is the Gumbel-max trick: for uniform `r`, `y_t` is an exact sample from `p_t`, so `q̄_t = p_t`. **Single-text tests are mathematically impossible** (Test A yields nothing).

**Leak: determinism.** Same context + key → same token. Normal sampling at T>0 is diverse; this isn't.

- **Test D — diversity collapse:** query the same prompt `n` times; compare the number of distinct outputs to what the model's entropy predicts. Collapsed diversity at nonzero temperature reveals the scheme.
- **Cache-augmented variants** (watermark disabled for already-seen contexts, to hide determinism) leak differently: first-query vs repeat-query behaviour differs.

Gloaguen et al. (2024/2025) built rigorous tests for seven variants across Red-Green, Fixed-Sampling and Cache-Augmented families, found both distribution-modifying and distribution-preserving schemes easily detectable in restricted black-box settings, and ran them cheaply against GPT-4, Claude 3 and Gemini 1.0 Pro.

---

## 5. Family 3 — Cryptographically undetectable schemes

Christ, Gunn & Zamir (2023): seed the PRF only after enough entropy has accumulated, so contexts essentially never repeat. They prove any efficient distinguisher without the key would break the PRF. **Keyless detection is provably impossible** under standard crypto assumptions. Publicly-verifiable variants (Fairoze et al., signature via rejection sampling) claim the same: distortion-free and undetectable without the public key.

---

## 6. Summary

| What you have              | Red-Green (KGW)                    | Gumbel / fixed-sampling         | Cryptographic   |
|----------------------------|------------------------------------|---------------------------------|-----------------|
| One text, no model         | No                                 | No                              | No              |
| One text + proxy model     | Weak (confounded with temperature) | No (provably)                   | No              |
| Large corpus + proxy       | Yes (bigram residuals)             | Sometimes (repeated contexts)   | No              |
| Query access to model      | Yes, cheaply                       | Yes (diversity collapse)        | No (provably)   |

**Practical answer:** deployed-style schemes tend to leak through a statistical side channel and are detectable without the key given enough samples or queries.
**Theoretical answer:** a well-designed scheme can be made undetectable; detection then reduces to breaking cryptography.

Recent work: TTP-Detect (March 2026) frames third-party verification as a relative hypothesis test, using a proxy model to amplify watermark-relevant signals.

---

## 7. Proposed experiment (for Claude Code)

Use a small open model (GPT-2 or a small Qwen) with Hugging Face `transformers`.

1. **Implement a KGW watermark** as a `LogitsProcessor`: hash `(prev_token, key)` → seed → green list of size `γ·|V|`; add `δ` to green logits. Parameters to sweep: `γ ∈ {0.25, 0.5}`, `δ ∈ {1, 2, 4}`.
2. **Implement a Gumbel/Aaronson watermark** as a custom sampler (PRF over a context window of `h` tokens).
3. **Generate datasets** from a fixed set of prompts:
   - plain sampling at T ∈ {0.7, 1.0, 1.2}
   - KGW-watermarked at T = 1.0
   - Gumbel-watermarked
4. **Keyed baseline:** confirm the KGW z-score detects the watermark (sanity check).
5. **Test A:** compute `Z` per text using the same model as the proxy, then a mismatched proxy. Plot `Z` distributions per condition. Expectation: KGW separates from T=1.0 plain, but overlaps with T=1.2 plain (the confound).
6. **Test B:** build the bigram residual matrix over the corpus; run a sign-consistency permutation test per preceding token. Expectation: separates KGW from all plain temperatures.
7. **Test D:** repeat identical prompts `n` times; count distinct outputs vs entropy-predicted expectation. Expectation: Gumbel collapses, plain does not.
8. **Report:** ROC curves / AUC per test per condition, and how detection power scales with text length `T` and corpus size.

Stretch goals: add a cache-augmented variant; try paraphrase attacks to measure robustness of each test.

---

## Sources

- Gloaguen, Jovanović, Staab, Vechev — *Black-Box Detection of Language Model Watermarks* (ICLR 2025; ICML 2024 FM-Wild workshop): https://arxiv.org/abs/2405.20777
- Fairoze et al. — *Publicly-Detectable Watermarking for Language Models*: https://arxiv.org/html/2310.18491v4
- Wang et al. — *Rethinking LLM Watermark Detection in Black-Box Settings: A Non-Intrusive Third-Party Framework* (TTP-Detect, 2026): https://sotaverified.org/papers/260314968
- Also referenced (not re-verified here): Kirchenbauer et al. 2023 (KGW); Kuditipudi et al. 2024 (robust distortion-free); Aaronson (Gumbel scheme); Christ, Gunn & Zamir 2023 (undetectable watermarks); Jovanović et al. 2024 (watermark stealing); Gehrmann et al. 2019 (GLTR).
