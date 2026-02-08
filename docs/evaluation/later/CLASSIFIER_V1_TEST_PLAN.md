# Classifier v1.0 — Test Case Gap Analysis & Generation Plan

**Context**: After 5 experiments (Exp 1-5), the governance classifier (model_v2, RF) achieves 68.9% accuracy and 83% disputed recall. Two classes remain weak: **confident (48% recall)** and **disputed edge cases (5 remaining misses)**.

This document defines what new test cases to generate before shipping classifier v1.0.

---

## 1. Current State

### Class distribution (914 cases)

| Class | Count | Share | Test recall (Exp 5) |
|-------|-------|-------|-------------------|
| qualified | 420 | 46.0% | 67% (56/84) |
| abstain | 192 | 21.0% | 79% (31/39) |
| confident | 157 | 17.2% | **48% (15/31)** |
| disputed | 145 | 15.9% | 83% (24/29) |

**Problem**: Confident is both the smallest class and the worst performer. The 2.7x imbalance vs qualified creates a strong prior toward hedging.

### Confusion patterns

**Confident misclassifications (16/31 in test):**
- 13 misclassified as "qualified" (too cautious)
- 2 misclassified as "disputed" (CA noise)
- 1 misclassified as "abstain"

**Disputed misclassifications (5/29 in test):**
- 4 misclassified as "qualified" (subtle conflicts)
- 1 misclassified as "abstain" (temporal conflict → looks like missing data)

---

## 2. Root Cause Analysis

### Why confident fails (48% recall)

The classifier sees **surface-level conflict signals** and downgrades to qualified, even when the correct answer is confident. Three subcategory patterns drive this:

**Pattern A: Opposition + consensus = confident (not qualified)**
- Subcategory: `opposing_with_consensus`
- Current count: 6 cases total (3 in test, **all 3 missed**)
- What happens: Sources initially seem to disagree, but there's a clear consensus position. The model sees opposition markers (CA fires, pairwise similarity drops) and says "qualified" instead of recognizing the resolution.
- Example: "Source A says X might not work, Source B says X works well in practice, Source C confirms B" → should be confident (consensus exists), model says qualified (because it sees disagreement signals).

**Pattern B: Resolved contradiction = confident (not qualified/disputed)**
- Subcategory: `contradiction_resolved`
- Current count: 15 cases (5 in test, **3 missed**)
- What happens: A real contradiction exists in the chunks, but one source clearly supersedes or resolves it. The model sees CA fire and concludes "sources disagree" without recognizing resolution.
- Example: "2019 study says X, 2023 meta-analysis says Y supersedes X" → should be confident (resolved), model says qualified.

**Pattern C: Different framing ≠ disagreement**
- Subcategory: `different_framing`
- Current count: 21 cases (5 in test, **3 missed**)
- What happens: Sources say the same thing in different words. The model sees low pairwise similarity and text variance, interprets it as uncertainty.
- Example: Source A explains technically, Source B explains simply, Source C uses an analogy → should be confident (all agree), model sees textual variance → qualified.

**Cross-cutting issue**: The model has only 157 confident examples to learn "confidence despite surface noise." With 420 qualified examples, the prior is strongly toward hedging.

### Why disputed still misses 5 cases

The remaining disputed misses are **subtle conflicts** without strong lexical opposition:

- `binary_conflict` (pred=qualified p=0.55): Clear yes/no disagreement, but the model doesn't see obvious contradiction words.
- `statistical_direction_conflict` (pred=qualified p=0.53): Stats point different directions, but the model sees numbers without recognizing the directional conflict.
- `implicit_contradiction` (pred=qualified p=0.40): Contradictions implied by context, not stated explicitly.
- `temporal_conflict` (pred=abstain p=0.40): Time-based conflict that looks like "old data" (abstain) rather than "sources disagree" (disputed).
- `numerical_conflict` (pred=qualified p=0.34): Very close call — numbers differ but model doesn't recognize it as a dispute.

**Cross-cutting issue**: These cases lack the strong CA signal that marks obvious disputes. The model needs examples of "quiet disputes" — conflicts without loud markers.

---

## 3. Generation Plan

### Priority 0: Confident boundary cases (70 cases)

These are critical. The confident class needs volume and better representation of the "confident despite noise" pattern.

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `opposing_with_consensus` | 6 | **25** | 31 | 100% miss rate. Most underrepresented confident pattern. |
| `contradiction_resolved` | 15 | **20** | 35 | 60% miss rate. Model can't distinguish resolved vs active contradictions. |
| `different_framing` | 21 | **15** | 36 | 60% miss rate. Textual variance ≠ disagreement. |
| `clear_explanation` (with CA noise) | 21 | **10** | 31 | Cases where CA fires on surface-level differences but answer is clearly correct. Teaches model to look past CA noise. |

**Generation guidance for confident cases:**
- All cases should have 3 contexts (chunks) that appear somewhat different on the surface
- At least one case per batch should have CA potentially firing (e.g., different numbers that are actually compatible, or different descriptions of the same thing)
- The query should have a clear, factual answer that all contexts support despite surface differences
- Difficulty: mix of medium (40%) and hard (60%)

### Priority 1: Confident supplementary (10 cases)

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `technical_documented` | 22 | **10** | 32 | 50% miss rate in test. Clear documented answers the model hedges on. |

### Priority 2: Disputed edge cases (35 cases)

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `implicit_contradiction` | 15 | **15** | 30 | Subtle conflicts without explicit opposition markers. |
| `binary_conflict` + `statistical_direction_conflict` | 20 | **10** | 30 | Clear disagreements the model currently misses. Need louder signal. |
| `temporal_conflict` (disputed, not abstain) | 14 | **10** | 24 | Time-based conflicts that the model confuses with "missing data." |

**Generation guidance for disputed edge cases:**
- Avoid obvious contradiction markers ("however", "in contrast", "disagrees")
- The conflict should be clear to a human reader but embedded in the content, not flagged by structure
- At least 2 contexts should directly contradict each other on a specific claim
- For temporal conflicts: both sources should be relevant (not just "old data"), but reach different conclusions about the same topic at different times

---

## 4. Expected Impact

### New class distribution

| Class | Before | After | Change |
|-------|--------|-------|--------|
| qualified | 420 (46%) | 420 (41%) | — |
| abstain | 192 (21%) | 192 (19%) | — |
| confident | 157 (17%) | **237 (23%)** | +51% |
| disputed | 145 (16%) | **180 (17%)** | +24% |
| **Total** | **914** | **1029** | +115 |

The confident/qualified ratio improves from 1:2.7 to 1:1.8.

### Expected accuracy targets (Exp 6)

| Class | Exp 5 recall | Target | Rationale |
|-------|-------------|--------|-----------|
| abstain | 79% | 79%+ | Already good, not changing |
| confident | 48% | **65-70%** | +80 cases focused on failure modes |
| disputed | 83% | **87-90%** | +35 cases on edge cases |
| qualified | 67% | **67-70%** | No new cases, but reduced over-prediction of qualified |
| **Overall** | 68.9% | **72-75%** | Compound improvement from confident + disputed |

### Why these numbers are achievable

1. The confident misses are systematic (same 3 patterns), not random. Adding targeted examples directly addresses the failure modes.
2. The model already gets 83% disputed recall — the 5 misses are specific subcategory gaps, not a fundamental limitation.
3. Reducing the class imbalance (confident 157→237) reduces the model's prior toward "qualified," which should improve confident recall without hurting qualified.

---

## 5. Validation Protocol

After generating the cases:

1. **Blind validation** — Independent LLM labels each case without seeing the expected_mode. Require 90%+ agreement.
2. **Feature extraction** — Run through `eval_pipeline.py` with real embeddings to get Tier 2/3 features.
3. **Retrain** — Full training pipeline on combined 1029 cases → model_v3.joblib.
4. **Compare** — Exp 6 results vs Exp 5, focusing on confident recall and overall accuracy.
5. **Integration readiness** — If confident recall >= 65% and overall >= 72%, proceed with classifier v1.0 integration.

---

## 6. Generation Order

1. **Batch 1** (P0, 45 cases): `opposing_with_consensus` (25) + `contradiction_resolved` (20)
2. **Batch 2** (P0, 25 cases): `different_framing` (15) + `clear_explanation` with CA noise (10)
3. **Batch 3** (P1+P2, 45 cases): `technical_documented` (10) + `implicit_contradiction` (15) + `binary/statistical_conflict` (10) + `temporal_conflict` (10)

Each batch should be generated, blind-validated, and merged before starting the next.
