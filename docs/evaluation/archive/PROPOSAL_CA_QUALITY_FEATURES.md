# Proposal: Contradiction Quality Features for Governance Classifier

**Status**: FAILED — fast LLM (qwen2.5:3b) cannot produce meaningful scores
**Priority**: High (addresses 49% of all classifier errors)
**Date**: 2026-02-10

---

## Problem

The governance classifier's #1 error pattern is **trustworthy cases misclassified as disputed** (32 of 65 errors = 49%). Root cause: `ca_fired=True` for 70% of trustworthy cases AND 98% of disputed cases. The binary signal can't distinguish real contradictions from superficial ones.

```
ca_fired=True breakdown (706 cases):
  disputed:    192 (27%)  <- real contradictions
  trustworthy: 477 (68%)  <- false alarms (different-framing, resolved, etc.)
  abstain:      37 (5%)
```

Current CA features are almost entirely uniform:
- `ca_signal`: always "disputed" when fired (zero info)
- `ca_first_evidence_char`: 87% "assertive" (near-constant)
- `ca_evidence_characters`: 86% "assertive_vs_assertive" (near-constant)

The fast LLM says "CONTRADICT" or "AGREE" — a binary answer that loses all nuance about contradiction *strength*, *type*, and *resolution*.

## Analysis

### Stage 2 feature ceiling

| Model | Stage 2 CV accuracy |
|-------|---------------------|
| RF default | 74.3% |
| RF regularized | 73.1% |
| GBT | 70.4% |
| GBT tuned | 70.6% |
| GBT top-10 features | 70.7% |

Switching algorithms, regularizing, and feature selection all converge to ~74%. **The ceiling is in the features, not the model.**

### Mutual information analysis (Stage 2)

Top features that actually discriminate disputed vs trustworthy:
1. `ca_fired` (0.085) — best single feature, but very noisy
2. `ctx_length_mean` (0.080) — text length proxy
3. `score_spread` (0.065) — vector score variance

10 features have MI=0.000 for Stage 2 (pure noise).

### Cohen's d within ca_fired=True (disputed vs trustworthy)

| Feature | Disputed | Trustworthy | Cohen's d |
|---------|----------|-------------|-----------|
| ctx_total_chars | 1243 | 1591 | 0.461 |
| score_spread | 0.110 | 0.084 | 0.315 |
| ca_pairs_checked | 1.005 | 1.063 | 0.273 |

Best separation is only d=0.46 (small-medium effect). The features overlap heavily.

### Error profile (32 trustworthy->disputed cases)

- All 32 have `ca_fired=True` — the CA constraint fired on superficial differences
- 20 were originally `qualified` (natural uncertainty markers look like disputes)
- 12 were originally `confident` (contradiction_resolved, entity_ambiguity subcategories)
- The model is moderately confident but wrong: P(trustworthy)=0.414 mean

## Proposed Solution: Richer CA Output

Replace the binary CONTRADICT/AGREE LLM response with a structured response that captures:

### 1. Contradiction score (0-10)

Instead of asking "Do these contradict? YES/NO", ask "How strongly do these contradict? 0-10" in the same LLM call. This gives a continuous signal the classifier can threshold on.

### 2. Contradiction type classification

Different contradiction types have very different rates of being "real" disputes:
- **numerical_conflict**: numbers disagree — often real disputes
- **opposing_conclusions**: opposite claims — usually real disputes
- **different_framing**: same info, different emphasis — usually NOT disputes
- **methodology_difference**: different approaches — context-dependent
- **temporal_conflict**: different time periods — often NOT disputes

### 3. Derived features for classifier

From score_all_pairs() across all chunk pairs, compute:
- `ca_max_score`: highest contradiction score across all pairs (0-10)
- `ca_mean_score`: average contradiction score across checked pairs
- `ca_score_spread`: max - min score (high spread = one real contradiction among noise)
- `ca_contradiction_type`: most common type seen

## Implementation

### New prompt (replaces binary CONTRADICT/AGREE)

```
Score how strongly these texts contradict each other about the question.

Question: {query}
Text 1: {text1}
Text 2: {text2}

SCORE: 0 = fully compatible, 5 = some tension, 10 = direct contradiction

TYPE (pick one):
- numerical: different numbers/stats for same metric
- opposing: opposite conclusions or claims
- temporal: different time periods cause difference
- framing: same underlying info, different emphasis
- methodology: different approaches/methods lead to different results
- compatible: no real contradiction

Reply in format: SCORE: N TYPE: word
```

### Where it runs

Option A (simple): Run the scoring prompt on ALL pairs in `apply()`, replacing the binary check. The constraint still short-circuits on first contradiction (score >= threshold) for deny/allow, but stores the scores in metadata for the classifier.

Option B (selective — from Two-Tier plan): Only run scoring on uncertain classifier cases. This is more complex and deferred.

**Chosen: Option A** — simpler, and the additional LLM cost per pair is zero (we're replacing the existing binary call, not adding a new one).

### Integration points

1. `conflict_aware.py` — Change `_check_pairwise_contradiction()` to return `(bool, float, str)` instead of `bool`. Store scores in `ca_diag`.
2. `feature_extractor.py` — Extract `ca_max_score`, `ca_mean_score`, `ca_score_spread`, `ca_contradiction_type` from metadata.
3. `eval_pipeline.py` — Add new features to CSV output.
4. `train_classifier.py` — Add new features to feature lists.

### Expected impact

The 32 trustworthy->disputed errors should shrink significantly because:
- "Different framing" pairs will get low scores (2-3/10) and type=framing
- "Resolved contradictions" will get low scores with type=compatible
- Real disputes will get high scores (7-10) with type=numerical/opposing

If scores provide a gap of even d=0.5 between disputed and trustworthy, that's a substantial improvement over the current d=0.0 binary signal.

### Cost

Zero additional LLM calls — we're changing the prompt format, not adding calls. The response goes from 1 word ("CONTRADICT") to ~5 words ("SCORE: 3 TYPE: framing"), which is negligible.

## Verification plan

1. Run scoring prompt on 20 disputed + 20 trustworthy cases manually
2. Check score gap: target mean_disputed - mean_trustworthy > 3.0
3. Check type distribution: trustworthy should cluster on framing/compatible/temporal
4. Re-extract full dataset with new features
5. Retrain two-stage model, compare Stage 2 CV to baseline (74.3%)
6. Target: Stage 2 CV > 78% (would lift combined accuracy to ~85%)

## Results: FAILED (2026-02-09)

### Scoring prompt test

Implemented the full scoring prompt and tested with qwen2.5:3b (fast LLM):

| Test case | Expected | Score returned | Type returned |
|-----------|----------|---------------|---------------|
| Compatible (Paris facts) | 0-2 | **10** | framing |
| Mild tension | 3-5 | **10** | framing |
| Real contradiction | 7-10 | **10** | opposing |
| Temporal difference | 3-5 | **10** | framing |
| Direct opposition | 8-10 | **10** | opposing |

**All scores are 10.** The 3B model cannot distinguish between "fully compatible" and "direct contradiction" on a 0-10 scale.

### Type-only classification test

Simplified to just type classification ("Pick ONE word: numerical, opposing, temporal, framing, compatible"):

| Actual type | Predicted type | Correct? |
|-------------|---------------|----------|
| numerical | opposing | No |
| opposing | opposing | Yes |
| framing | compatible | No |
| temporal | opposing | Yes/No (edge) |

**1/4 correct.** The 3B model defaults to "opposing" for most inputs.

### Root cause

Small LLMs (3B params) can handle binary yes/no decisions but lack calibration for:
1. Scalar scoring on continuous scales (always gives maximum)
2. Fine-grained categorical classification (defaults to most salient category)

### Implication

The "zero additional LLM cost" assumption is invalid. Getting richer CA features requires either:
1. A smarter LLM (14B+) — the Two-Tier approach from the plan
2. Deterministic text features (no LLM) — overlap ratios, number extraction, sentiment
3. More training data in the hardest subcategories

All changes reverted.
