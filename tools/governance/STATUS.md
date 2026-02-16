# Governance Classifier Status

**Date**: 2026-02-16
**Branch**: feature/krag-addition

## Architecture

Two-stage classifier:
- **Stage 1** (XGBClassifier): abstain vs answerable (binary). Threshold: 0.710
- **Stage 2** (StackingClassifier: XGB+GBT+LGBM): trustworthy vs disputed (binary). Threshold: 0.585
- S2 only trained on answerable cases (trustworthy + disputed). **S2 never sees abstain data.**

Files:
- `fitz_ai/governance/decider.py` — production inference
- `fitz_ai/governance/constraints/feature_extractor.py` — feature extraction at query time
- `tools/governance/train_classifier.py` — training + calibration
- `tools/governance/extract_features.py` — batch feature extraction from fitz-gov cases
- `tools/governance/data/features.csv` — 2920 rows x 82 columns
- `tools/governance/data/model_v5_twostage.joblib` — trained model (pre-calibration thresholds)
- `tools/governance/data/model_v5_calibrated.joblib` — calibrated model (production)

## CRITICAL BUG: S2 Blind Spot on Leaked Abstain

### The Problem

When S1 incorrectly predicts an abstain case as "answerable" (p_answerable >= 0.710), that case
flows to S2. But S2 was only trained on answerable data (trustworthy + disputed), so it has no
concept of "abstain." It confidently predicts these leaked cases as trustworthy.

Numbers from 5-fold CV:
- S1 leaks 104/685 abstain cases (15.2%)
- Of those, S2 predicts 89/104 as trustworthy (85.6%)
- Leaked abstain p_answerable: median=0.873 (NOT borderline — spread across full range)
- Leaked abstain S2 p_trustworthy: median=0.876 (confidently wrong)

### What Was Tried (and Failed)

1. **Train S2 on all data (trustworthy vs not-trustworthy)**: 157 FT total, worse than baseline.
   Reason: disputed and abstain have very different signal patterns; lumping them muddied both.

2. **Add S1 p(answerable) as S2 feature**: 175 FT total, even worse.
   Reason: S2 only trains on answerable cases where s1_p_answerable is always high; can't learn
   what low values mean.

3. **Borderline zone (force disputed if p_answerable near threshold)**: Doesn't work because
   leaked abstain cases aren't near the threshold — median p_answerable is 0.873, max is 1.0.

### What Was Done Instead (COSMETIC FIX ONLY)

In `train_classifier.py` calibration, abstain cases get `p_trustworthy = 0.0` so they always
map to "disputed" in the evaluation sweep. This makes calibration report "0 FT-abstain" but
**does NOT fix the production behavior**. In `decider.py`, leaked abstain cases go through S2
and get real (high) p_trustworthy values → FT-abstain in production.

### Actual Production FT Count (Honest)

With current thresholds (s1=0.710, s2=0.585):
- FT-abstain: **89** (not 0 as calibration claims)
- FT-disputed: 63
- Total FT: **152** (not 63 as calibration claims)
- Still better than governor's 411, but significantly worse than reported

## Honest Benchmark (5-fold CV, n=2920)

### Best Honest Operating Point (s1=0.795, s2=0.680)

| Metric | Value |
|--------|-------|
| Accuracy | 72.3% |
| Total FT | 95 (61 abstain + 34 disputed) |
| Abstain recall | 89.5% |
| Disputed recall | 90.8% |
| Trustworthy recall | 56.7% |

### Current Thresholds (s1=0.710, s2=0.585) — Real Production Numbers

| Metric | Value |
|--------|-------|
| Accuracy | 78.0% |
| Total FT | 152 (89 abstain + 63 disputed) |
| Abstain recall | 84.8% |
| Disputed recall | 87.7% |
| Trustworthy recall | 70.8% |

### Governor Baseline

| Metric | Value |
|--------|-------|
| Accuracy | 56.7% |
| Total FT | 411 (354 abstain + 57 disputed) |
| Abstain recall | 40.7% |
| Disputed recall | 84.2% |
| Trustworthy recall | 51.8% |

## The Core Tradeoff

Lowering FT requires raising thresholds, which tanks trustworthy recall:
- 152 FT → 70.8% trustworthy recall (current)
- 95 FT → 56.7% trustworthy recall (best safe)
- 0 FT → ~0% trustworthy recall (everything is disputed/abstain)

## Feature Consistency Issues

### decider.py Missing Features

`decider.py` has incomplete `_CATEGORICAL_FEATURES` and `_BOOL_FEATURES` sets compared to
`train_classifier.py`. Missing from decider:

**Categoricals** (3 missing): `ie_detection_reason`, `ie_query_aspect`, `query_question_type`
- Impact: These get passed as numeric 0 instead of label-encoded values at inference

**Bools** (15 missing): `av_fired`, `ca_is_uncertainty_query`, `detection_aggregation`,
`detection_boost_authority`, `detection_boost_recency`, `detection_needs_rewriting`,
`has_abstain_signal`, `has_disputed_signal`, `ie_critical_match_found`, `ie_entity_match_found`,
`ie_has_conflicting_aspect`, `ie_has_matching_aspect`, `ie_primary_match_found`,
`ie_summary_overlap`, `query_has_comparison_words`
- Impact: String "True"/"False" values won't be mapped to 0/1 correctly

### Interaction Features (16 missing from production)

The model uses 16 `ix_*` interaction features computed in `train_classifier.py` during training
(e.g., `ix_ca_x_vector`, `ix_av_votes_x_vector`, `ix_divergence_no_ca`). These are NOT computed
in `decider.py` or `feature_extractor.py`. At inference, they're all 0.

`ix_divergence_no_ca` was #10 in S1 feature importance (0.0235).

## Untracked Files (cleanup candidates)

- `nul` — Windows artifact, delete
- `run_extraction.bat` — one-off batch script, delete
- `tmp_test_sklearn.py` — temp test, delete
- `tools/governance/ANALYSIS_SUMMARY.md` — analysis notes
- `tools/governance/analyze_regression.py` — ad-hoc analysis script
- `tools/governance/constraint_ignored_analysis.py` — ad-hoc analysis
- `tools/governance/deep_dive_analysis.py` — ad-hoc analysis
- `tools/governance/visualize_findings.py` — ad-hoc analysis

## What Needs To Happen

1. **Fix the S2 blind spot in production** — Options:
   - (a) Add safety check in `decider.py`: if p_answerable < some_safe_threshold, cap output to disputed
   - (b) Recalibrate with honest FT numbers and accept higher thresholds (s1=0.795, s2=0.680)
   - (c) Train a 3-class S2 (trustworthy/disputed/abstain) instead of binary
   - (d) Add a third "abstain detector" stage after S2
   - Option (a) doesn't work because leaked abstain cases aren't borderline (median p_ans=0.873)
   - Option (b) works but trustworthy recall drops to 56.7%
   - Options (c) and (d) are untried

2. **Sync decider.py feature sets** — Add missing categoricals, bools to match train_classifier.py

3. **Add interaction feature computation** to either `feature_extractor.py` or `decider.py`

4. **Clean up untracked files**

5. **Re-run calibration honestly** once a real fix is in place

## Recent Commit History

```
c290ca6 [fix] Fix calibration safety: vectorized sweep, safe abstain fallback  <-- COSMETIC FIX ONLY
7f466ff [update] Add numerical divergence features to governance classifier
e26fcee [update] Add safety-focused threshold calibration to reduce false-trustworthy
1fc8ddf [update] Improve governance classifier: interaction features for abstain detection
```
