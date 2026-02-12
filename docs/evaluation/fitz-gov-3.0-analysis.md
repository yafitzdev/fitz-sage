# fitz-gov 3.0: Technical Analysis

**Purpose**: How we got from 26.9% (rules) to 90.9% (two-stage ML) — what worked, what failed, and why.
**Related**: [fitz-gov-3.0-results.md](fitz-gov-3.0-results.md) for the summary of what was achieved.

---

## Phase 1: Why Rules Failed

The v2.0 governor used priority rules over constraint outputs:

```
IE abstain signal    -> ABSTAIN   (highest priority)
dispute signal       -> DISPUTED
any denial           -> QUALIFIED
nothing fired        -> CONFIDENT (lowest priority)
```

On the expanded 1113-case dataset, this scored **26.9%**. Three fundamental problems:

1. **Can't handle signal interactions** — dispute signal + high relevance + hedging language might mean "qualified", but rules always pick "disputed"
2. **Over-predicts after tuning** — tighter CA prompts (Exp 3) improved dispute detection but made the governor predict "disputed" for 60% of cases
3. **No learning** — rules can't improve from labeled data

The classifier sees the same constraint outputs as *features* and learns decision boundaries from data.

---

## Phase 2: 4-Class Classifier (Experiments 1-7)

### The Approach

Constraints become feature extractors instead of decision-makers. ~50 numeric/categorical features feed a tabular classifier that predicts one of 4 governance modes (abstain/disputed/qualified/confident).

### Results

| Experiment | Model | Accuracy | Notes |
|------------|-------|----------|-------|
| Exp 1 | GBT | 57.4% | Baseline, 47 features |
| Exp 2 | RF | 71.0% | +context features, +class weighting |
| Exp 5 | RF | 68.9% | Retrained on real features; disputed 83% |
| **Exp 6** | **GBT** | **69.1%** | **+199 cases (1113 total). Best 4-class.** |
| Exp 7a | GBT | 66.8% | +6 text features (noise, reverted) |
| Exp 7b | GBT | 60.1% | Longer hyperparam search (worse) |

### Why 69.1% Was the Ceiling

The model was secretly a 2-class classifier. It only predicted abstain + qualified, achieving **0% recall on both confident and disputed**. The features could not separate confident from qualified.

**Feature quality analysis** revealed the problem:

| Feature | Split Importance (rank) | Permutation Importance (rank) |
|---------|------------------------|------------------------------|
| `ctx_length_mean` | 0.129 (#1) | 0.090 (#1) |
| `has_disputed_signal` | 0.052 (#5) | **0.001 (#27)** |
| `ca_signal` | top 15 | **not in top 30** |

Constraint signals appeared important in split-based rankings but contributed near-zero actual accuracy when permuted. The classifier relied on context length proxies instead. 10 features were constant zero, 8 were redundant (r > 0.95), leaving ~30 effective features out of 50.

**Class separability**: Maximum correlation between confident and qualified features was r=0.23. The distinction was meaningful to humans but invisible to the feature set.

---

## Phase 3: The 3-Class Pivot

### Collapsing Confident + Qualified

Since the 4-class model couldn't separate confident from qualified, we collapsed them into **trustworthy**. The question changed from "how confident is the system?" to "can I trust this answer?"

Production still distinguishes them: if constraints fired, the trustworthy answer gets caveats (qualified behavior); if nothing fired, it answers directly (confident behavior). The distinction moved from the *classifier* to the *response generation* layer.

| Metric | 4-class (Exp 6) | 3-class GBT |
|--------|-----------------|-------------|
| Test accuracy | 69.1% | **72.7%** |
| Abstain recall | 60% | **72.9%** |
| Disputed recall | ~0% | **28.2%** |
| Trustworthy recall | n/a | **85.3%** |

The 3-class model actually learns all three classes — disputed went from 0% to 28.2%. But 28.2% is still unacceptable.

---

## Phase 4: Two-Stage Binary Classifiers

### Why Decomposition Works

A single 3-class classifier wastes splits. Stage 1 doesn't need to distinguish disputed from trustworthy; Stage 2 doesn't need to filter abstain. Binary classifiers get cleaner signal.

```
Stage 1: Can we answer?  (ExtraTrees)
    P(answerable) < threshold -> ABSTAIN
    P(answerable) >= threshold -> proceed

Stage 2: Do sources agree?  (RandomForest)
    P(trustworthy) < threshold -> DISPUTED
    P(trustworthy) >= threshold -> TRUSTWORTHY
```

### Different Feature Profiles

Each stage learned different things:

**Stage 1** (abstain vs answerable): `ca_fired`, `has_disputed_signal`, `ca_signal`, `query_word_count`, `mean_vector_score` — constraint signals dominate, because "can we answer at all?" depends on whether the context is relevant.

**Stage 2** (trustworthy vs disputed): `ctx_length_mean`, `ctx_total_chars`, `ctx_length_std`, `ctx_mean_pairwise_sim`, `score_spread`, `chunk_length_cv` — context proxy features dominate, because "do sources agree?" requires comparing the evidence.

### Threshold Tuning

The classifiers output probabilities. Thresholds determine the safety/recall tradeoff.

Early calibration (s1=0.50, s2=0.70):

| Metric | Raw (0.5/0.5) | Calibrated (0.5/0.7) |
|--------|---------------|----------------------|
| Disputed recall | 53.9% | **76.9%** (+23.1pp) |
| Trustworthy recall | 91.9% | 81.6% (-10.3pp) |

Lowering the Stage 2 trustworthy threshold forces uncertain predictions toward "disputed" — trading unnecessary hedging (annoying but harmless) for catching real conflicts (critical for safety).

---

## Failed Approaches

### Continuous CA Scoring (Steps 2/2b)

**Hypothesis**: Replace binary CONTRADICT/AGREE prompt with continuous 0-10 scoring to give the classifier richer signal.

**Result**: The fast LLM (qwen2.5:3b) **always returned "SCORE: 10"** regardless of content. Even clearly compatible texts got maximum contradiction scores. A follow-up test with type-only classification (numerical/opposing/temporal/framing/compatible) scored 1/4 correct.

**Root cause**: 3B-parameter models lack calibration for scalar scoring. They handle binary yes/no decisions but cannot distinguish "slight tension" from "direct contradiction."

Both attempts regressed (67.3%, 65.5%). Fully reverted.

### New Text Features (Exp 7a)

Added 6 features: hedging count/ratio, assertive count/ratio, unique number count, exclusive numbers ratio.

**Result**: Dropped from 69.1% to 66.8%. Features correlated with context length (already #1 by importance) and added noise without orthogonal signal.

### Extended Hyperparameter Search (Exp 7b)

Tripled search budget from 200s to 600s per model.

**Result**: 60.1%. The longer search explored a different hyperparameter region and converged on a shallow model (max_depth=2) that can't capture feature interactions. RandomizedSearchCV is stochastic — more iterations can find different (worse) local optima.

---

## Successful Approaches

### 1. Inter-Chunk Deterministic Features (Proposal 1b)

Since the fast LLM can't provide scoring nuance, we derived discrimination signal from deterministic text analysis. No LLM calls needed.

5 new features, computed from raw chunk text:

| Feature | Cohen's d (Stage 2, ca_fired=True) |
|---------|------|
| `chunk_length_cv` | **0.424** |
| `max_pairwise_overlap` | 0.146 |
| `min_pairwise_overlap` | 0.139 |
| `number_density` | 0.114 |
| `assertion_density` | 0.006 |

`chunk_length_cv` was the strongest new discriminator. Disputed cases have significantly higher chunk length variance (mean 0.181 vs 0.120 for trustworthy when CA fired). Intuitively: when sources have very different sizes, they're more likely from different contexts, increasing contradiction risk.

**Impact**: Stage 2 CV improved +10.5pp (74.3% -> 84.8%), total test errors dropped 28% (65 -> 47).

### 2. Feature Parity Fix (Proposal 2)

**Problem**: The top 3 Stage 2 features by importance (`ctx_length_mean`, `ctx_total_chars`, `ctx_length_std`) only existed at training time in `train_classifier.py:compute_context_features()`. They were NOT in production `feature_extractor.py`. The model would silently degrade in production because its most important features would all be zero at inference time.

**Fix**: Ported 12 features from training code to production feature extractor. Added TF-IDF cosine similarity features, contradiction markers, negation counts, numerical variance, temporal features.

**Impact**: Accuracy unchanged (expected — training features were already correct), but the train/inference gap was closed. Without this, production would have been broken.

### 3. Dead Feature Cleanup

Audit found 18 removable features (38% of 47): 10 constant zero, 8 redundant (r > 0.95), 4 near-constant. Removed 826 lines of dead code including 2 unused plugin files. Retrained on 29 clean features — identical accuracy, confirming the dead features were truly dead.

### 4. The Big Win: Embedding Distribution Fix

**Root cause**: `extract_features.py` never computed embeddings. Training data had `mean_vector_score=0`, `std_vector_score=0`, `score_spread=0` for ALL 1113 cases. The eval pipeline computed real embeddings, creating a massive train/eval distribution mismatch.

**Fix**: Added ollama embedder + DetectionOrchestrator to `extract_features.py`. Re-extracted all 1113 cases with real embeddings and detection features.

**Impact**:

| Phase | Abstain | Disputed | Trustworthy | Overall |
|-------|---------|----------|-------------|---------|
| Before (Feb 10) | 81.2% | 89.7% | 70.6% | 76.5% |
| **After (Feb 11)** | **93.7%** | **94.4%** | **89.0%** | **90.9%** |
| Delta | +12.5pp | +4.7pp | +18.4pp | +14.4pp |

The entire improvement came from feeding the model the features it was supposed to have. Same architecture, same constraints, same data — just correct feature computation.

---

## Feature Inventory (Final State)

50 features across 4 tiers:

| Tier | Count | Source | Examples |
|------|-------|--------|----------|
| Constraint metadata | ~15 | Guardrail outputs | `ca_fired`, `ie_fired`, `has_disputed_signal`, `evidence_character` |
| Vector/retrieval scores | ~8 | VectorSearchStep | `mean_vector_score`, `score_spread`, `chunk_count` |
| Context text analysis | ~20 | Deterministic from chunks | `ctx_length_mean`, `chunk_length_cv`, `ctx_mean_pairwise_sim` |
| Detection flags | ~7 | DetectionOrchestrator | `detection_temporal`, `detection_comparison`, `vocab_overlap_ratio` |

All features are available at inference time with no additional LLM calls. Constraints already run before the governance decision; context features are computed from raw chunk text; detection runs during retrieval.

---

## Path Forward

The 15 remaining critical cases (false trustworthy) are all hard-difficulty boundary cases:

- **9 abstain -> trustworthy**: Wrong entity/version/domain with high vector overlap (decoy keywords). The entity mismatch is invisible to current features.
- **6 disputed -> trustworthy**: Implicit contradictions with low lexical similarity. CA didn't fire because the contradiction isn't stated explicitly.

All 15 have `ie_fired=False`. Average `p_answerable=0.752` (barely clearing the s1=0.55 gate) vs 0.858 for correct trustworthy predictions.

**What would help**:
- Entity-mismatch detector (query mentions entity X, context discusses entity Y)
- Chunk-sufficiency check (single chunk with high similarity but incomplete answer)
- Richer CA signals (requires better LLM than 3B, or multi-source test cases)

These are constraint-level improvements, not threshold tuning. The classifier has reached the ceiling of what current features can provide.
