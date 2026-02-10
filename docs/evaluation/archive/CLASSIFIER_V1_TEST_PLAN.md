# Classifier v1.0 — Test Case Gap Analysis & Generation Plan

**Status**: EXECUTED (Experiment 6). 199 cases generated, validated at 93.5%, merged. Results: GBT 69.1% overall (below predicted 73-76%), confident 62% (below 65% target), disputed 67% (below predicted 87-90%). Shipped as model_v3.joblib despite targets not being met — see NOTEPAD.md Exp 6-7.

**Context**: After 5 experiments (Exp 1-5), the governance classifier (model_v2, RF) achieves 68.9% accuracy and 83% disputed recall. Two classes remain weak: **confident (48% recall)** and **disputed edge cases (5 remaining misses)**.

This document defined what new test cases to generate before shipping classifier v1.0.

---

## 1. Current State

### Class distribution (914 cases)

| Class | Count | Share | Test recall (Exp 5) |
|-------|-------|-------|-------------------|
| qualified | 420 | 46.0% | 67% (56/84) |
| abstain | 192 | 21.0% | 79% (31/39) |
| confident | 157 | 17.2% | **48% (15/31)** |
| disputed | 145 | 15.9% | 83% (24/29) |

**Problems**:
1. Confident is both the smallest class and the worst performer.
2. The 2.9:1 ratio between largest (qualified) and smallest (disputed) creates strong class prior toward hedging.
3. Abstain, while performing well, could regress as other classes grow.

### Confusion patterns

**Confident misclassifications (16/31 in test):**
- 13 misclassified as "qualified" (too cautious)
- 2 misclassified as "disputed" (CA noise)
- 1 misclassified as "abstain"

**Disputed misclassifications (5/29 in test):**
- 4 misclassified as "qualified" (subtle conflicts)
- 1 misclassified as "abstain" (temporal conflict looks like missing data)

---

## 2. Root Cause Analysis

### Why confident fails (48% recall)

The classifier sees **surface-level conflict signals** and downgrades to qualified, even when the correct answer is confident. Three subcategory patterns drive this:

**Pattern A: Opposition + consensus = confident (not qualified)**
- Subcategory: `opposing_with_consensus`
- Current count: 6 cases total (3 in test, **all 3 missed**)
- What happens: Sources initially seem to disagree, but there's a clear consensus position. The model sees opposition markers (CA fires, pairwise similarity drops) and says "qualified" instead of recognizing the resolution.
- Example: "Source A says X might not work, Source B says X works well in practice, Source C confirms B" -> should be confident (consensus exists), model says qualified (because it sees disagreement signals).

**Pattern B: Resolved contradiction = confident (not qualified/disputed)**
- Subcategory: `contradiction_resolved`
- Current count: 15 cases (5 in test, **3 missed**)
- What happens: A real contradiction exists in the chunks, but one source clearly supersedes or resolves it. The model sees CA fire and concludes "sources disagree" without recognizing resolution.
- Example: "2019 study says X, 2023 meta-analysis says Y supersedes X" -> should be confident (resolved), model says qualified.

**Pattern C: Different framing != disagreement**
- Subcategory: `different_framing`
- Current count: 21 cases (5 in test, **3 missed**)
- What happens: Sources say the same thing in different words. The model sees low pairwise similarity and text variance, interprets it as uncertainty.
- Example: Source A explains technically, Source B explains simply, Source C uses an analogy -> should be confident (all agree), model sees textual variance -> qualified.

**Cross-cutting issue**: The model has only 157 confident examples to learn "confidence despite surface noise." With 420 qualified examples, the prior is strongly toward hedging.

### Why disputed still misses 5 cases

The remaining disputed misses are **subtle conflicts** without strong lexical opposition:

- `binary_conflict` (pred=qualified p=0.55): Clear yes/no disagreement, but the model doesn't see obvious contradiction words.
- `statistical_direction_conflict` (pred=qualified p=0.53): Stats point different directions, but the model sees numbers without recognizing the directional conflict.
- `implicit_contradiction` (pred=qualified p=0.40): Contradictions implied by context, not stated explicitly.
- `temporal_conflict` (pred=abstain p=0.40): Time-based conflict that looks like "old data" (abstain) rather than "sources disagree" (disputed).
- `numerical_conflict` (pred=qualified p=0.34): Very close call -- numbers differ but model doesn't recognize it as a dispute.

**Cross-cutting issue**: These cases lack the strong CA signal that marks obvious disputes. The model needs examples of "quiet disputes" -- conflicts without loud markers.

---

## 3. Generation Plan — 200 New Cases

**Strategy**: Add 200 cases across the 3 underrepresented classes to reduce max:min class ratio from 2.9:1 to 2.0:1, while focusing volume on the weakest failure modes.

### Target distribution

| Class | Before | Add | After | Share |
|-------|--------|-----|-------|-------|
| qualified | 420 (46%) | 0 | 420 (37.7%) | -- |
| confident | 157 (17%) | **+95** | **252 (22.6%)** | largest gain |
| abstain | 192 (21%) | **+45** | **237 (21.3%)** | stability buffer |
| disputed | 145 (16%) | **+60** | **205 (18.4%)** | edge case focus |
| **Total** | **914** | **+200** | **1114** | |
| **Max:min ratio** | **2.9:1** | | **2.0:1** | |

### Confident cases (+95)

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `opposing_with_consensus` | 6 | **30** | 36 | 100% miss rate. Most underrepresented confident pattern. |
| `contradiction_resolved` | 15 | **25** | 40 | 60% miss rate. Model can't distinguish resolved vs active contradictions. |
| `different_framing` | 21 | **20** | 41 | 60% miss rate. Textual variance != disagreement. |
| `clear_explanation` (with CA noise) | 21 | **10** | 31 | CA fires on surface-level differences but answer is clearly correct. |
| `technical_documented` | 22 | **10** | 32 | 50% miss rate. Clear documented answers the model hedges on. |

**Generation guidance for confident cases:**
- All cases should have 3 contexts (chunks) that appear somewhat different on the surface
- At least one case per batch should have CA potentially firing (e.g., different numbers that are actually compatible, or different descriptions of the same thing)
- The query should have a clear, factual answer that all contexts support despite surface differences
- Difficulty: mix of medium (40%) and hard (60%)

### Disputed cases (+60)

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `implicit_contradiction` | 15 | **20** | 35 | Subtle conflicts without explicit opposition markers. Biggest disputed gap. |
| `binary_conflict` | 15 | **10** | 25 | Clear yes/no disagreements the model currently misses (p=0.55 for qualified). |
| `statistical_direction_conflict` | 5 | **10** | 15 | Stats point different directions; severely underrepresented. |
| `temporal_conflict` | 14 | **10** | 24 | Time-based conflicts confused with "missing data" (abstain). |
| `numerical_conflict` (boundary) | 26 | **10** | 36 | Close numerical differences that are real conflicts, not rounding. |

**Generation guidance for disputed cases:**
- Avoid obvious contradiction markers ("however", "in contrast", "disagrees")
- The conflict should be clear to a human reader but embedded in the content, not flagged by structure
- At least 2 contexts should directly contradict each other on a specific claim
- For temporal conflicts: both sources should be relevant (not just "old data"), but reach different conclusions
- For numerical/statistical: use realistic domain data where the numbers matter (financial, medical, engineering)

### Abstain cases (+45)

| Subcategory | Current | Add | Target | Rationale |
|-------------|---------|-----|--------|-----------|
| `wrong_entity` | 26 | **10** | 36 | Strengthen the most common abstain pattern |
| `off_topic_contradiction` | 20 | **10** | 30 | Contexts contradict but are irrelevant to query |
| `temporal_mismatch` | 13 | **10** | 23 | Query asks about time period not covered by contexts |
| `wrong_specificity` | 13 | **8** | 21 | Query is more specific than contexts can answer |
| `cross_domain_insufficient` | 3 | **7** | 10 | Severely underrepresented. Contexts from wrong domain. |

**Generation guidance for abstain cases:**
- The query should be clear and reasonable, but the provided contexts genuinely cannot answer it
- Avoid cases that could be argued as "qualified" -- the evidence gap should be unambiguous
- For wrong_entity/domain: contexts should be topically adjacent enough to seem relevant at first glance
- For temporal_mismatch: contexts should be clearly from the wrong time period for what the query asks

---

## 4. Expected Impact

### Expected accuracy targets (Exp 6)

| Class | Exp 5 recall | Target | Rationale |
|-------|-------------|--------|-----------|
| abstain | 79% | **80-82%** | +45 cases provide stability, prevent regression |
| confident | 48% | **65-70%** | +95 cases focused on 3 failure patterns |
| disputed | 83% | **87-90%** | +60 cases on subtle/edge conflicts |
| qualified | 67% | **67-70%** | No new cases, but reduced over-prediction from rebalancing |
| **Overall** | 68.9% | **73-76%** | Compound improvement from all three classes |

### Why these numbers are achievable

1. The confident misses are systematic (same 3 patterns), not random. Adding targeted examples directly addresses the failure modes.
2. The model already gets 83% disputed recall -- the 5 misses are specific subcategory gaps, not a fundamental limitation.
3. Reducing the class imbalance (max:min from 2.9:1 to 2.0:1) reduces the model's prior toward "qualified."
4. Abstain cases prevent regression as the dataset grows -- maintaining the anchor class.

---

## 5. Validation Protocol

After generating the cases:

1. **Blind validation** -- Independent LLM labels each case without seeing the expected_mode. Require 90%+ agreement.
2. **Feature extraction** -- Run through `eval_pipeline.py` with real embeddings to get Tier 2/3 features.
3. **Retrain** -- Full training pipeline on combined 1114 cases -> model_v3.joblib.
4. **Compare** -- Exp 6 results vs Exp 5, focusing on confident recall and overall accuracy.
5. **Ablation test** -- Run without context features (ctx_*) to verify the model doesn't collapse. If it does, add regularization.
6. **Leave-one-subcategory-out CV** -- Hold out entire subcategories to test generalization beyond memorized patterns.
7. **Integration readiness** -- If confident recall >= 65% and overall >= 73%, proceed with classifier v1.0 integration.

---

## 6. Generation Order

1. **Batch 1** (55 cases): `opposing_with_consensus` (30) + `contradiction_resolved` (25) -- highest priority confident patterns
2. **Batch 2** (30 cases): `different_framing` (20) + `clear_explanation` with CA noise (10) -- remaining confident gaps
3. **Batch 3** (30 cases): `implicit_contradiction` (20) + `binary/statistical_conflict` (10) -- disputed edge cases
4. **Batch 4** (40 cases): `temporal_conflict` (10) + `numerical_conflict` (10) + `technical_documented` (10) + `temporal_mismatch/wrong_specificity/cross_domain` (10) -- disputed + abstain
5. **Batch 5** (45 cases): remaining abstain subcategories (35) + remaining disputed (10) -- balance completion

Each batch should be generated, blind-validated, and merged before starting the next.
