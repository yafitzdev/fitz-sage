# Classifier v1.0 — Next Steps for Quality Improvement

**Current state**: GBT classifier, 69.1% overall accuracy on 1113 cases.
**Target**: 70%+ recall in every category.

---

## Current Per-Class Performance

| Class | Recall | Target | Gap | Difficulty |
|-------|--------|--------|-----|------------|
| Abstain | 85% | 70%+ | Met | Easy — clear signals (IE fires, entity mismatch) |
| Disputed | 67% | 70%+ | -3pp | Medium — CA fires but subtle conflicts get missed |
| Qualified | 66% | 70%+ | -4pp | Hard — catch-all class, 18 subcategories, every boundary |
| Confident | 62% | 70%+ | -8pp | Hard — must learn "absence of problems" |

Disputed and qualified are close. Confident needs the most work.

---

## Implementation Order

Do these in sequence. Each builds on the previous.

### Step 1: Per-Class Calibrated Thresholds — DONE

**Result**: +0.9pp accuracy (69.1% → 70.0%), +1.9pp min recall (61.5% → 63.5%).

Swept all 4-class threshold combinations via grid search. Optimal thresholds: abstain=0.60, confident=0.50, disputed=0.00, qualified=0.00. Only 5/223 test cases (2.2%) fall back to governor. On those 5, classifier was 0/5 correct, governor was 2/5.

**Why limited impact**: The governor is only 27% accurate overall (massively over-predicts disputed). Fallback to a bad governor can't help much. As the governor improves, this infrastructure becomes more valuable — every fallback case automatically benefits from better rules with no retraining.

**Per-class results**:

| Class | Before | After | Delta |
|-------|--------|-------|-------|
| Abstain | 85.1% | 85.1% | +0.0pp |
| Confident | 61.5% | 63.5% | +1.9pp |
| Disputed | 66.7% | 69.2% | +2.6pp |
| Qualified | 65.9% | 65.9% | +0.0pp |

**Files**: `tools/governance/calibrate_thresholds.py`, `tools/governance/data/model_v3_calibrated.joblib`

### Step 2: Continuous CA Signals (the #1 feature unlock)

**The problem**: `has_disputed_signal` and `ca_signal` are binary (0/1). A marginal "these sources kinda disagree" gets the same weight as "these sources flatly contradict each other." This is why CA ranks #5-6 in feature importance instead of #1 — binary features have fewer split opportunities than continuous ones.

**New features to extract from CA**:

| Feature | Type | Source | What it captures |
|---------|------|--------|-----------------|
| `ca_contradiction_confidence` | float [0-1] | LLM response parsing | How confident is the LLM that a contradiction exists? |
| `ca_contradiction_density` | float [0-1] | Pairwise results | Proportion of chunk pairs flagged as contradicting |
| `ca_max_contradiction_score` | float [0-1] | Per-pair confidence | Strength of the strongest contradiction |
| `ca_mean_contradiction_score` | float [0-1] | Per-pair confidence | Average contradiction strength across all pairs |
| `ca_conflicting_chunk_ratio` | float [0-1] | Pairwise results | Proportion of chunks involved in at least one contradiction |

**Implementation**: CA already does pairwise LLM comparisons. Currently it parses the response as "yes/no contradiction." Instead:
1. Ask the LLM to rate contradiction confidence (0-10 scale) in the existing prompt
2. Store per-pair scores instead of just the final boolean
3. Compute density/max/mean as aggregate features

**Expected impact**: Once CA stops being a boolean, it should outrank context length features. This directly targets the 13 missed disputed cases (subtle conflicts where CA fires weakly or not at all).

**Effort**: Modify CA constraint prompt + result parsing (~2h), re-extract features (~15min), retrain (~15min).

### Step 3: Source Agreement Features (positive confident signals)

**The problem**: Confident is detected by the *absence* of problems — no constraint fires, so it must be confident. This is fragile. We need *positive* evidence for confidence: the sources actually agree.

**New features**:

| Feature | Type | Source | What it captures |
|---------|------|--------|-----------------|
| `source_agreement` | bool | LLM call | "Do these passages reach the same conclusion?" |
| `source_agreement_confidence` | float [0-1] | LLM response | How strongly do they agree? |
| `semantic_variance_agreeing` | float | Embedding similarity | Low variance + agreement = strong confident signal |
| `distinct_conclusions` | int | LLM call | Number of distinct conclusions across contexts |
| `agreement_after_resolution` | bool | CA + agreement | Sources agree once temporal/authority ordering is applied |

**Implementation**: 1 additional LLM call per case during feature extraction:
```
"Do these N passages reach the same conclusion about [query]?
Rate agreement: 0 (completely disagree) to 10 (identical conclusions).
How many distinct conclusions are present?"
```

**Expected impact**: +3-5pp confident recall. This directly targets the 15 confident→qualified failures where the classifier over-hedges because it has no positive signal for confidence.

**Effort**: New LLM prompt + feature extraction (~3h), retrain (~15min). Cost: +1 LLM call per case at both training and inference time.

### Step 4: Governor-Classifier Hybrid as Product Feature

**The reframe**: The governor fallback from Step 1 isn't a hack — it's **defensive governance**. Make it a first-class product feature with three explicit decision tiers:

| Tier | Condition | Label | Meaning |
|------|-----------|-------|---------|
| **ML confident** | `max_proba >= class_threshold` | `mode: disputed (ml, p=0.82)` | Classifier is confident in its prediction |
| **Rule-based override** | `max_proba < threshold` AND governor disagrees | `mode: qualified (rule-override, p=0.38)` | Classifier unsure, governor's rules applied |
| **Low-confidence fallback** | `max_proba < threshold` AND governor agrees | `mode: qualified (low-conf, p=0.41)` | Both systems agree but neither is confident |

**The governance mode becomes a tuple**: `(mode, decision_source, confidence)` instead of just `mode`.

**Why this matters**:
- **Auditors** can trace every decision to its source
- **Users** see calibrated confidence, not just a label
- **Developers** can tune per-tier behavior independently
- **Monitoring** can track which tier handles what fraction of queries — if rule-based overrides increase, the classifier is degrading

**Implementation**: Wrap classifier + governor in a `GovernanceDecider` class that:
1. Runs constraints (feature extraction + governor decision)
2. Runs classifier on features
3. Compares confidence to per-class threshold
4. Returns `GovernanceDecision(mode, source, confidence, features)`

**Effort**: Integration work (~4h). No ML changes — this is pure architecture.

---

### Step 5: Cascade Classifier for Uncertain Cases

**The insight**: Instead of falling back to the governor (27% accuracy) when the classifier is uncertain, train a second specialist classifier on the uncertain cases. The first classifier handles the easy 95%+, the second learns the hard boundary cases.

**Prerequisites**: Exhaust Steps 2-4 first. Each of those directly improves the primary classifier's features, which is higher-leverage than adding a second model. The cascade is most useful after the main model's ceiling is truly hit.

**Design**:
1. Set thresholds to route ~15-25% of cases to the second stage (not the current 2.2%)
2. Train a specialist on the low-confidence cases, possibly with different features or model type
3. The specialist could use subcategory as a feature (risky for the main model, but the specialist sees a narrower problem)

**Open question**: Artificially raising thresholds to get more second-stage training data is tempting but risks overfitting. With 223 test cases and 15-25% routed = ~35-55 cases for the specialist — marginal for a 4-class problem. Cross-validation on the full dataset might work better than a single held-out split. Alternatively, the specialist could be a binary classifier for just the confused pairs (e.g., "confident vs qualified" and "disputed vs qualified") rather than a full 4-class model.

**Blocked on**: Governor improvement (Step 4) and primary classifier improvements (Steps 2-3). Revisit after those are exhausted.

### Step 6: Split Qualified into Sub-Classes

**The insight**: Qualified is the chaos class — 18 subcategories, 360 cases, every boundary touches it. The classifier treats "methodology_difference" and "hedged_claims" identically even though they have very different constraint signatures.

**Potential split** (needs validation):

| Sub-class | Subcategories | Signal pattern |
|-----------|---------------|---------------|
| **Evidence quality** | hedged_claims, small_sample, source_quality, source_quality_asymmetry | IE fires but evidence is weak |
| **Scope mismatch** | different_aspects, same_claim_different_conditions, methodology_difference, conditional_applicability | CA might fire but contexts aren't really contradicting |
| **Temporal caveat** | temporal_ambiguity, evolving_facts, same_claim_different_timeperiods, deprecated_documented | Detection temporal + freshness signals |
| **Partial answer** | partial_answer, related_missing_specific, right_topic_wrong_infotype | IE fires weakly, SIT fires |

**Risk**: Splitting reduces per-class sample sizes (360/4 = ~90 each, with some sub-classes having <20 cases). The classifier might overfit. Also, the product still needs a single "qualified" mode — the split is purely internal to improve classification, then map back to qualified for the user.

**Approach**: First analyze whether the sub-classes actually have different feature distributions (box plots of key features per sub-class). If they overlap heavily, the split won't help. If they cluster clearly, the multi-class problem becomes easier.

**Blocked on**: Requires subcategory analysis tooling. Lower priority than Steps 2-4.

---

## Additional Improvements (Lower Priority)

### Feature Engineering

**A. Pairwise embedding similarity**
- Current: TF-IDF pairwise similarity between contexts.
- Better: Cosine similarity between context embeddings (the actual vectors used in retrieval).
- This directly measures "how similar do the vector DB think these chunks are?" — a stronger signal than TF-IDF.

**B. Query-context embedding distribution**
- Add per-chunk vector score distribution features: skewness, kurtosis, percentiles.
- Abstain cases should have uniformly low scores. Disputed should have high scores with high variance.

**C. LLM confidence signals**
- Extract the LLM's own uncertainty markers from constraint responses — hedging language in CA verdicts, jury vote margins in AV.
- Meta-learning: using the LLM's behavior as a feature.

### Disputed Recall

**A. Numerical magnitude feature**
- Current: `ca_numerical_variance_detected` is binary (within 25% or not).
- Add: How far apart are the most divergent numbers? Continuous feature.

**B. Temporal conflict detector**
- Detect if contexts reference different time periods for the same claim.
- Regex for year patterns + check if multiple years appear in different contexts.

**C. Assertion polarity feature**
- Extract polarity of each context's main claim (positive/negative/neutral).
- Simple negation detection per context + comparison.

### Qualified Recall

**A. Subcategory-aware training**
- Add subcategory as a feature (label-encoded). Gives the model a hint about the type of qualification.
- Risk: Might overfit to subcategory distribution.

**B. Hedging language features**
- Tried in Exp 7a — hedging counts correlated with context length and added noise.
- Retry with hedging *ratio* only (normalized by length) or with hedging *density* (hedging words / sentence count).
- The ratio might be orthogonal enough to help where raw counts failed.

**C. Evidence completeness feature**
- Does the query ask for something specific that the contexts don't contain?
- SIT constraint already detects this partially. Generalize it.

### Training Improvements

**A. Cross-validation instead of single split**
- Currently: Single 80/20 split, seed=42.
- Better: Report mean +/- std across 5 stratified folds.

**B. Per-class optimization**
- Custom scoring function: optimize for minimum per-class recall, not overall accuracy.

**C. Feature selection / RFE**
- 58 features may include noise.
- Hypothesis: Removing proxy features (ctx_length_*) might force the model to learn from governance signals.

### Data Quality

**A. Real-world failure collection** (highest long-term value)
- Once integrated, collect cases where the classifier disagrees with human judgment.
- These are the highest-value training examples — real failures, not synthetic ones.

**B. Active learning**
- Log all predictions with probabilities.
- Cases where max_proba is near the decision boundary (0.3-0.5) are the most informative to label next.

**C. Audit small subcategories**
- `code_abstention` (3 cases), `cross_domain_insufficient` (3 cases), `source_conflict` (4 cases).
- These contribute noise, not signal.

---

## Priority Summary

| Step | Action | Target | Expected Impact | Effort | Retraining? | Status |
|------|--------|--------|----------------|--------|-------------|--------|
| **1** | **Per-class calibrated thresholds** | All | +0.9pp accuracy, +1.9pp min recall | **Low** | No | **DONE** |
| **2** | **Continuous CA signals** | Disputed | +3-5pp disputed | **Low-Medium** | Yes | Next |
| **3** | **Source agreement features** | Confident | +3-5pp confident | **Medium** | Yes | |
| **4** | **Governor-classifier hybrid** | All | Architectural (also improves Step 1 fallback) | **Medium** | No | |
| **5** | **Cascade classifier** | Low-confidence cases | +2-4pp on uncertain cases | **Medium** | Yes (2nd model) | Blocked on 2-4 |
| **6** | **Split qualified sub-classes** | Qualified | +2-5pp qualified | **Medium** | Yes | Needs analysis |
| 7 | Pairwise embedding similarity | Disputed/Confident | +1-2pp | Medium | Yes | |
| 8 | Cross-validation + per-class optimization | All | +2-3pp | Low | Yes | |
| 9 | Real-world failure collection | All | Compound | Ongoing | Yes | |
| 10 | Feature selection / RFE | All | +1-2pp | Medium | Yes | |

**Steps 2-4 are the critical path.** Step 1 is done (+0.9pp). Steps 2-3 target the weakest classes directly with better features. Step 4 makes the governor better (which retroactively improves Step 1's fallback). Steps 5-6 are structural changes to attempt after the feature-level improvements are exhausted.
