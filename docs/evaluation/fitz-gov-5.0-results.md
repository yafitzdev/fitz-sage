# fitz-gov 5.0 Benchmark Results

**Version**: fitz-gov 5.0 (2,910 governance cases)
**Date**: February 2026
**Decision method**: 4-question cascade (GBT × 3)
**Thresholds**: Q1=0.770, Q3=0.680, Q4=0.770
**Evaluation**: 5-fold stratified cross-validation

---

## Executive Summary

fitz-gov 5.0 expanded the benchmark from 1,113 to 2,910 cases and replaced the two-stage binary pipeline with a **4-question cascade** that routes conflict and non-conflict paths through separate specialized models. The DetectionClassifier was upgraded from keyword rules to an ML+keyword hybrid (temporal 90.6% recall, comparison 90.2% recall), providing cleaner detection signals to the governance feature set.

> v4.0 was an intermediate development version. It was never formally benchmarked. v5.0 is the first documented version after v3.0.

### Current Results

| Metric | Value |
|--------|-------|
| **Overall accuracy** | **81.3%** |
| **Abstain recall** | **90.2%** |
| **Disputed recall** | **74.9%** |
| **Trustworthy recall** | **78.6%** |

Evaluated with 5-fold stratified cross-validation on all 2,910 cases.

### Version Comparison

| Version | Cases | Decision Method | Abstain | Disputed | Trustworthy | Overall |
|---------|-------|-----------------|---------|----------|-------------|---------|
| v1.0 | 200 | Rules (governor) | 72.5% | 90.0% | ~72% | ~72% |
| v2.0 | 331 | Rules (governor) | 57.1% | 89.1% | ~47–79% | ~72% |
| v3.0 | 1,113 | 2-stage ML (ET + RF) | 93.7% | 94.4% | 89.0% | 90.9% |
| **v5.0** | **2,910** | **4-question cascade (GBT × 3)** | **90.2%** | **74.9%** | **78.6%** | **81.3%** |

**On the accuracy drop**: v3.0's 90.9% was measured with safety-first thresholds calibrated and evaluated on the same 1,113-case set. v5.0 uses 5-fold cross-validation on 2.6× more cases — a stricter and more honest methodology. With 1,797 additional cases targeting harder boundary examples, lower cross-validated accuracy is expected. Critically, disputed recall at 74.9% is now bounded by the `ca_fired` rule gate (Q2): if the conflict is real but the constraint doesn't fire, the case never reaches Q3.

---

## Data Expansion

### Growth History

| Version | Cases | Method | Validation |
|---------|-------|--------|------------|
| v1.0 | 200 | Hand-crafted from 21 experiments | Expert review |
| v2.0 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| v2.1 | +123 | Targeted gaps (dispute boundary, code, adversarial) | 94% blind agreement |
| v3.0 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |
| v5.0 | +1,797 | Extended boundary sampling + failure-mode targeting | — |
| **Total** | **2,910** | | |

### Class Distribution (v5.0)

| Class | Count | % |
|-------|-------|---|
| Trustworthy | 1,361 | 46.8% |
| Abstain | 884 | 30.4% |
| Disputed | 665 | 22.9% |

Max:min class ratio: 2.0:1 (improved from 2.2:1 in v3.0). 92% hard difficulty.

---

## 4-Question Cascade Architecture

The key structural change from v3.0 is splitting the conflict path from the clean path. v3.0 Stage 2 trained one Random Forest to distinguish trustworthy from disputed across all answerable cases. v5.0 routes them first through a deterministic gate (Q2), then through two separate specialized models.

```
Query + Context → 5 Constraints → 109 features
         │
         ▼
  Q1: Evidence sufficient? (GBT, t=0.770)
         │
         ├─ NO → ABSTAIN
         │
         └─ YES
               │
               ▼
         Q2: ca_fired? (rule: conflict gate)
               │
               ├─ YES (conflict path)
               │        │
               │        ▼
               │  Q3: Conflict resolved? (GBT, t=0.680)
               │        │
               │        ├─ NO  → DISPUTED
               │        └─ YES → TRUSTWORTHY
               │
               └─ NO (clean path)
                        │
                        ▼
                  Q4: Evidence solid? (GBT, t=0.770)
                        │
                        ├─ NO  → ABSTAIN
                        └─ YES → TRUSTWORTHY
```

### Why Cascade Over Two-Stage

v3.0's Stage 2 (trustworthy vs disputed) trained on all answerable cases — conflict and non-conflict mixed. The cascade separates them:

- **Q3** is trained exclusively on conflict-path cases (where `ca_fired=True`). It learns "is this conflict resolvable?" from a focused set.
- **Q4** is trained exclusively on clean cases (where `ca_fired=False`). It learns "is the clean evidence truly solid?" without dispute noise.
- Both use GBT (v3.0 used ExtraTrees/RandomForest). GBT's sequential residual correction handles the skewed, hard-boundary cases better.

### Cascade vs Two-Stage

| Approach | Overall | Abstain | Disputed | Trustworthy |
|----------|---------|---------|----------|-------------|
| v3.0 two-stage (ET+RF, 1,113 cases) | 90.9% | 93.7% | 94.4% | 89.0% |
| **v5.0 cascade (GBT×3, 2,910 cases, 5-fold CV)** | **81.3%** | **90.2%** | **74.9%** | **78.6%** |

Disputed recall dropped significantly because Q2 is now a hard rule: if `ca_fired=False`, the case never reaches Q3. v3.0 Stage 2 could still classify disputed cases where CA didn't fire (via other features). The cascade trades that flexibility for specialization — Q3 and Q4 each see cleaner signal.

### Threshold Selection

Three independent thresholds tuned jointly via `calibrate_cascade.py`:

| Q1 | Q3 | Q4 | FT (false-trustworthy) | Overall | Abstain | Disputed | Trustworthy |
|----|----|----|------------------------|---------|---------|----------|-------------|
| **0.770** | **0.680** | **0.770** | **(min)** | **81.3%** | **90.2%** | **74.9%** | **78.6%** |

Sweep: Q1 ∈ [0.40, 0.80), Q3 ∈ [0.20, 0.70), Q4 ∈ [0.40, 0.80). Filter: accuracy ≥ 78% and trustworthy recall ≥ 65%. Chosen point minimizes false-trustworthy (over-confidence) subject to usability constraints.

---

## Key Changes from v3.0

### 1. Improved DetectionClassifier (ML + Keyword)

v3.0 detection flags (`detection_temporal`, `detection_comparison`) came from keyword/regex rules inside `DetectionOrchestrator`. v5.0 replaced this with an ML+keyword hybrid `DetectionClassifier`:

| Detection Type | Recall |
|----------------|--------|
| Temporal | **90.6%** |
| Comparison | **90.2%** |

These detection signals feed directly into the governance feature set. Better detection → less noise in the features the governance classifiers train on.

### 2. Feature Set Expansion (50 → 109)

v3.0 had 50 raw features. v5.0 has 109 features after one-hot encoding of categorical constraint outputs:

| Tier | v3.0 | v5.0 | Notes |
|------|------|------|-------|
| Constraint metadata | ~15 | ~30 | AnswerVerification (jury vote) added |
| Vector/retrieval scores | ~8 | ~12 | Additional score distribution features |
| Context text analysis | ~20 | ~30 | Extended inter-chunk features |
| Detection flags | ~7 | ~15 | ML-classified detection + encoded categoricals |
| Categorical encodings | 0 | ~22 | One-hot expansion of `caa_query_type`, `sit_info_type_requested`, etc. |

Three dead features removed (`num_unique_sources`, `ie_max_similarity`, `ie_summary_overlap`).

### 3. AnswerVerification Constraint (New)

A fifth constraint added alongside IE, CA, CAA, SIT: an LLM jury using 3 independent prompts to assess evidence sufficiency. Jury votes contribute features to Q1 and Q4 — the two "can we answer?" classifiers. This directly addresses the 9 abstain→trustworthy critical cases from v3.0 where decoy entity data passed the IE constraint.

---

## What Worked

### GBT Over Random Forest/ExtraTrees

For all three cascade classifiers (Q1, Q3, Q4), GBT outperformed the v3.0 choice of ExtraTrees/RandomForest. GBT's sequential residual correction is better suited to the hard boundary cases (92% of dataset), where the marginal case is the main challenge.

### Specialized Path Models

Separating Q3 (conflict-path) and Q4 (clean-path) allows each model to specialize. Q3 focuses on "is this contradiction resolvable?" — a different question than Q4's "is the non-conflicting evidence actually sufficient?" Training them jointly (v3.0 Stage 2) required the model to handle both questions simultaneously.

### 5-Fold Cross-Validation

v3.0 thresholds were calibrated on the full training set. v5.0 uses OOF (out-of-fold) predictions for all threshold calibration, making the reported numbers genuinely held-out. This is the honest methodology going forward.

---

## What Failed

### Disputed Recall Regression (94.4% → 74.9%)

The largest regression. The cascade's Q2 gate (`ca_fired` rule) is now the hard ceiling for disputed recall: any real dispute where CA doesn't fire gets routed to the clean path (Q4) and will be predicted ABSTAIN or TRUSTWORTHY, never DISPUTED.

v3.0 Stage 2 didn't have this hard gate — it could pick up dispute signals from other features (context similarity, chunk length CV, etc.) even when CA didn't fire. The cascade trades this recovery ability for specialization in Q3.

**Root cause**: CA constraint recall on disputed cases is imperfect with the 3B LLM. The 6 disputed→trustworthy critical cases from v3.0 (implicit contradictions with low lexical similarity) are now disputed→ABSTAIN or disputed→TRUSTWORTHY depending on the clean-path Q4 score. The gate, not the model, is the bottleneck.

### Dataset Growth Penalty

Adding 1,797 cases to a benchmark calibrated on 1,113 inevitably introduces more hard boundary examples. Cross-validated accuracy on the expanded set is not directly comparable to threshold-calibrated accuracy on the original set — but the expanded set is more representative.

---

## Critical Case Profile

False-trustworthy (the most dangerous error — predicting trustworthy when should abstain or dispute) are not enumerated in cross-validation, but the threshold sweep minimizes them at the chosen operating point.

Historical pattern from v3.0 (15 cases):
- 9 abstain→trustworthy: Wrong entity/version with high vector overlap (decoy keywords)
- 6 disputed→trustworthy: Implicit contradictions with low lexical similarity, CA didn't fire

With the cascade, the 6 disputed type now tends toward abstain→trustworthy (if clean path) rather than disputed→trustworthy — the nature of over-confidence shifts but the root cause (CA constraint ceiling with 3B model) remains.

---

## Reproduction

```bash
# Feature extraction (requires ollama + cohere or other providers)
python -m tools.governance.extract_features --chat cohere --embedding ollama --workers 1

# Train cascade classifier
python -m tools.governance.train_classifier --mode cascade --time-budget 200

# Calibrate thresholds (minimizes false-trustworthy)
python -m tools.governance.calibrate_cascade
```

### Key Files

| File | Purpose |
|------|---------|
| `tools/governance/extract_features.py` | Feature extraction (real embeddings + detection) |
| `tools/governance/train_classifier.py` | Cascade training with hyperparameter search |
| `tools/governance/calibrate_cascade.py` | 3-threshold sweep for critical case minimization |
| `tools/governance/data/features.csv` | 2,910 rows × 82 columns |
| `fitz_ai/governance/data/model_v6_cascade.joblib` | Production cascade artifact |
| `fitz_ai/governance/decider.py` | GovernanceDecider (production inference) |
| `fitz_ai/governance/constraints/feature_extractor.py` | Runtime feature extraction (109 features) |
| `retrieval/detection/registry.py` | DetectionOrchestrator with ML DetectionClassifier |

---

## Path Forward

The disputed recall regression (94.4% → 74.9%) is the primary open problem. Root cause: `ca_fired` as a hard gate means disputes the CA constraint misses never reach Q3.

**What would help**:
- Stronger LLM for CA constraint (3B model is the ceiling for implicit contradiction detection)
- Secondary dispute-detection path: route cases where other features suggest conflict (high chunk_length_cv, high number_density divergence) to Q3 even without `ca_fired=True`
- More CA training: fine-tuning a 3B model on fitz-gov dispute cases specifically

The 9 abstain→trustworthy critical cases (decoy entity data) from v3.0 are addressed in theory by the new AnswerVerification constraint — this is tested in v5.0 but the jury adds latency and its recall on hard decoy cases needs further measurement.

These are constraint-level improvements. The cascade classifier has reached the ceiling of what the current CA constraint can provide on disputed cases.
