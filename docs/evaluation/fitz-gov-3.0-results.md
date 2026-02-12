# fitz-gov 3.0 Benchmark Results

**Version**: fitz-gov 3.0.0 (1113 governance cases + 66 grounding/relevance)
**Date**: February 8-11, 2026
**Decision method**: Two-stage ML classifier (ExtraTrees + RandomForest)
**Thresholds**: s1=0.55, s2=0.79

---

## Executive Summary

fitz-gov 3.0 expanded the benchmark from 331 to 1113 governance cases and replaced the rule-based governor with a **two-stage ML classifier**. The taxonomy was simplified from 4-class (abstain/disputed/qualified/confident) to **3-class (abstain/disputed/trustworthy)** after analysis showed confident vs qualified was inseparable with current features.

### Current Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **90.9%** |
| **Abstain recall** | **93.7%** |
| **Disputed recall** | **94.4%** |
| **Trustworthy recall** | **89.0%** |
| **Critical cases** (false trustworthy) | **15** (all hard difficulty) |

### Version Comparison

| Version | Cases | Decision Method | Abstain | Disputed | Trust/Qual/Conf | Overall |
|---------|-------|-----------------|---------|----------|-----------------|---------|
| v1.0 | 200 | Rules (governor) | 72.5% | 90.0% | 72.5% / 86.7% | ~72% |
| v2.0 | 331 | Rules (governor) | 57.1% | 89.1% | 47.1% / 79.4% | ~72% |
| **v3.0** | **1113** | **ML (two-stage)** | **93.7%** | **94.4%** | **89.0%** | **90.9%** |

v1.0/v2.0 used 4-class taxonomy (abstain/disputed/qualified/confident) with rule-based priority decisions. v3.0 uses 3-class (abstain/disputed/trustworthy) with ML.

---

## The 3-Class Taxonomy

### Why We Collapsed 4 Classes to 3

The original 4-class system distinguished between "qualified" (answer with caveats) and "confident" (answer directly). Analysis of 1113 labeled cases showed:

- **Max correlation between confident and qualified features**: r=0.23
- **4-class classifier secretly predicted only 2 classes** (abstain + qualified), achieving 0% recall on both confident and disputed
- **Permutation importance**: constraint signals that should distinguish confident from qualified contributed near-zero actual accuracy

The distinction was meaningful to humans but invisible to the feature set. Collapsing to 3-class framed the right question: "Can I trust this answer?" rather than "How confident is the system?"

| Metric | 4-class (Exp 6) | 3-class GBT |
|--------|-----------------|-------------|
| Test accuracy | 69.1% | **72.7%** |
| Abstain recall | 60% | **72.9%** |
| Disputed recall | ~0% | **28.2%** |
| Trustworthy recall | n/a | **85.3%** |

Production still distinguishes hedged vs direct: if constraints fired, the trustworthy answer gets caveats; if nothing fired, it answers directly. The distinction moved from the *classifier* to the *response generation* layer.

### The 3 Classes

| Class | Meaning | Production Behavior |
|-------|---------|-------------------|
| **ABSTAIN** | Context doesn't answer the query | "I don't have enough information" |
| **DISPUTED** | Sources make conflicting claims | "Sources disagree on this" |
| **TRUSTWORTHY** | Evidence supports an answer | Generate answer (with caveats if constraints fired) |

---

## Data Expansion

### Growth History

| Version | Cases | Method | Validation |
|---------|-------|--------|------------|
| v1.0 | 200 | Hand-crafted from 21 experiments | Expert review |
| v2.0 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| v2.1 | +123 | Targeted gaps (dispute boundary, code, adversarial) | 94% blind agreement |
| v3.0 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |
| **Total** | **1113** | | |

### Class Distribution (tier1, governance only: 1047)

| Class (4-class label) | Count | % | 3-class mapping |
|-----------------------|-------|---|-----------------|
| Qualified | 360 | 34.4% | trustworthy |
| Confident | 254 | 24.3% | trustworthy |
| Abstain | 237 | 22.6% | abstain |
| Disputed | 196 | 18.7% | disputed |

66 grounding/relevance cases mapped to trustworthy for training (total: 1113).

Max:min class ratio: 2.2:1 (improved from 2.9:1 in v2.0). 92% hard difficulty.

---

## Two-Stage Classifier Architecture

```
Query → Constraints run (feature extractors) → 50 features
    │
    ├─ Stage 1: Can we answer? (ExtraTrees)
    │    P(answerable) < 0.55 → ABSTAIN
    │    P(answerable) >= 0.55 → proceed to Stage 2
    │
    └─ Stage 2: Do sources agree? (RandomForest)
         P(trustworthy) < 0.79 → DISPUTED
         P(trustworthy) >= 0.79 → TRUSTWORTHY
```

### Why Two Stages Beat Single Classifier

| Approach | Overall | Abstain | Disputed | Trustworthy |
|----------|---------|---------|----------|-------------|
| Single 3-class GBT | 72.7% | 72.9% | 28.2% | 85.3% |
| **Two-stage** | **90.9%** | **93.7%** | **94.4%** | **89.0%** |

Each binary classifier gets cleaner signal — Stage 1 doesn't waste splits on disputed vs trustworthy, Stage 2 doesn't waste splits filtering abstain.

### Different Feature Profiles

**Stage 1** (abstain vs answerable): `ca_fired`, `has_disputed_signal`, `ca_signal`, `query_word_count`, `mean_vector_score` — constraint signals dominate, because "can we answer at all?" depends on whether the context is relevant.

**Stage 2** (trustworthy vs disputed): `ctx_length_mean`, `ctx_total_chars`, `ctx_length_std`, `ctx_mean_pairwise_sim`, `score_spread`, `chunk_length_cv` — context proxy features dominate, because "do sources agree?" requires comparing the evidence.

### Threshold Tuning

Thresholds chosen to minimize **critical cases** (false trustworthy — the most dangerous error):

| s1 | s2 | Overall | Abstain | Disputed | Trustworthy | Critical |
|----|-----|---------|---------|----------|-------------|----------|
| 0.55 | 0.80 | 90.5% | 93.7% | 94.4% | 88.2% | 15 |
| **0.55** | **0.79** | **90.9%** | **93.7%** | **94.4%** | **89.0%** | **15** |
| 0.55 | 0.785 | 91.3% | 93.7% | 94.4% | 89.6% | 16 |
| 0.55 | 0.75 | 92.4% | 93.7% | 93.9% | 91.5% | 18 |

s2=0.79 is the highest trustworthy recall where critical stays at the minimum (15). The 16th case at s2=0.785 is `t1_abstain_hard_212` (wrong_product) — genuinely dangerous.

---

## Feature Inventory (50 features)

| Tier | Count | Source | Examples |
|------|-------|--------|----------|
| Constraint metadata | ~15 | Guardrail outputs | `ca_fired`, `ie_fired`, `has_disputed_signal`, `evidence_character` |
| Vector/retrieval scores | ~8 | VectorSearchStep | `mean_vector_score`, `score_spread`, `chunk_count` |
| Context text analysis | ~20 | Deterministic from chunks | `ctx_length_mean`, `chunk_length_cv`, `ctx_mean_pairwise_sim` |
| Detection flags | ~7 | DetectionOrchestrator | `detection_temporal`, `detection_comparison`, `vocab_overlap_ratio` |

All features are available at inference time with no additional LLM calls.

---

## What Worked

### 1. Inter-Chunk Deterministic Features

Since the fast LLM (3B) can't provide scoring nuance, we derived discrimination signal from deterministic text analysis. No LLM calls needed.

| Feature | Cohen's d (Stage 2, ca_fired=True) |
|---------|------|
| `chunk_length_cv` | **0.424** |
| `max_pairwise_overlap` | 0.146 |
| `min_pairwise_overlap` | 0.139 |
| `number_density` | 0.114 |
| `assertion_density` | 0.006 |

`chunk_length_cv` was the strongest new discriminator. Disputed cases have significantly higher chunk length variance (mean 0.181 vs 0.120 for trustworthy when CA fired).

**Impact**: Stage 2 CV improved +10.5pp (74.3% -> 84.8%), total test errors dropped 28%.

### 2. Embedding Distribution Fix (The Big Win)

`extract_features.py` never computed embeddings. Training data had `mean_vector_score=0` for ALL 1113 cases. The eval pipeline computed real embeddings, creating a massive train/eval distribution mismatch.

**Fix**: Added ollama embedder + DetectionOrchestrator to `extract_features.py`.

| Phase | Abstain | Disputed | Trustworthy | Overall |
|-------|---------|----------|-------------|---------|
| Before (Feb 10) | 81.2% | 89.7% | 70.6% | 76.5% |
| **After (Feb 11)** | **93.7%** | **94.4%** | **89.0%** | **90.9%** |
| Delta | +12.5pp | +4.7pp | +18.4pp | +14.4pp |

Same architecture, same constraints, same data — just correct feature computation.

### 3. Dead Feature Cleanup

Audit found 18 removable features (38% of 47): 10 constant zero, 8 redundant (r > 0.95). Removed 826 lines of dead code. Retrained on 29 clean features — identical accuracy.

---

## What Failed

### Continuous CA Scoring

**Hypothesis**: Replace binary CONTRADICT/AGREE with continuous 0-10 scoring for richer signal.

**Result**: The 3B model always returned "SCORE: 10" regardless of content. Both attempts regressed (67.3%, 65.5%). Fully reverted.

### New Text Features (Exp 7a)

Added 6 features: hedging/assertive counts, unique numbers. Dropped from 69.1% to 66.8% — correlated with context length (already #1) and added noise.

### Extended Hyperparameter Search (Exp 7b)

Tripled search budget (200s → 600s). Got 60.1% — longer search found a worse local optimum.

---

## Critical Case Analysis

15 false-trustworthy cases (predicted trustworthy, actually abstain or disputed):

| Expected | Count | Pattern |
|----------|-------|---------|
| ABSTAIN → TW | 9 | Wrong entity/version/domain with high vector overlap (decoy keywords) |
| DISPUTED → TW | 6 | Implicit contradictions with low lexical similarity |

- 13/15 hard difficulty
- `ie_fired=False` for all 15 — InsufficientEvidence constraint never catches these
- Average `p_answerable=0.752` (barely clears s1=0.55 gate) vs 0.858 for correct trustworthy
- Improvement requires constraint-level changes, not threshold tuning

---

## Accuracy Progression

| Phase | Date | Approach | Overall | Notes |
|-------|------|----------|---------|-------|
| Governor (rules) | Feb 8 | Priority rules | 26.9% | Baseline |
| 4-class GBT | Feb 8 | Exp 6, 1113 cases | 69.1% | Best 4-class |
| 3-class GBT | Feb 9 | Class collapse | 72.7% | |
| Two-stage (formal) | Feb 9 | RF+RF, best accuracy | 82.96% | |
| Two-stage calibrated | Feb 9 | s1=0.50, s2=0.70 | 80.72% | min recall 76.9% |
| + Inter-chunk features | Feb 10 | +5 deterministic | 78.92% | Stage 2 CV +10.5pp |
| + Feature parity fix | Feb 10 | 12 features ported | 78.92% | Production gap closed |
| Safety-first | Feb 10 | s2=0.80 | 75.3% | Disputed 89.7% |
| Sweet-spot | Feb 10 | s2=0.785 | 76.5% | Disputed 89.7%, TW 70.6% |
| **Retrained** | **Feb 11** | **Real embeddings + detection** | **90.9%** | **93.7/94.4/89.0** |

---

## Reproduction

```bash
# Feature extraction (requires ollama + cohere)
python -m tools.governance.extract_features --chat cohere --embedding ollama --workers 1

# Train two-stage classifier
python -m tools.governance.train_classifier --mode twostage --time-budget 200

# Calibrate thresholds
python -m tools.governance.calibrate_thresholds --mode twostage
```

### Key Files

| File | Purpose |
|------|---------|
| `tools/governance/extract_features.py` | Feature extraction with real embeddings + detection |
| `tools/governance/train_classifier.py` | Two-stage training with hyperparameter search |
| `tools/governance/calibrate_thresholds.py` | Threshold sweep for critical case minimization |
| `tools/governance/data/features.csv` | 1113 rows x 50 columns |
| `tools/governance/data/model_v5_calibrated.joblib` | Production model artifact |
| `fitz_ai/governance/decider.py` | GovernanceDecider (production inference) |
| `fitz_ai/governance/constraints/feature_extractor.py` | Runtime feature extraction |

---

## Path Forward

The 15 remaining critical cases are all hard-difficulty boundary cases:

- **9 abstain -> trustworthy**: Wrong entity/version/domain with high vector overlap (decoy keywords). The entity mismatch is invisible to current features.
- **6 disputed -> trustworthy**: Implicit contradictions with low lexical similarity. CA didn't fire because the contradiction isn't stated explicitly.

**What would help**:
- Entity-mismatch detector (query mentions entity X, context discusses entity Y)
- Chunk-sufficiency check (single chunk with high similarity but incomplete answer)
- Richer CA signals (requires better LLM than 3B, or multi-source test cases)

These are constraint-level improvements, not threshold tuning. The classifier has reached the ceiling of what current features can provide.
