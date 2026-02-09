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

### Step 1: Per-Class Calibrated Thresholds (fastest win, zero retraining)

**The insight**: Different classes have different error costs. A false confident is dangerous. A false qualified is annoying. The threshold should reflect that.

**What to do**:
- Load model_v3.joblib and the held-out test set
- Extract `predict_proba()` for all test cases
- Sweep per-class thresholds:
  - **Confident**: needs high confidence (`proba > 0.7`) — the cost of false confidence is dangerous
  - **Qualified**: can tolerate ambiguity (`proba > 0.4`) — over-hedging is safe
  - **Disputed**: should err early (`proba > 0.5`) — missing a conflict is worse than a false alarm
  - **Abstain**: keep default — already at 85%
- When no class exceeds its threshold, fall back to governor

**Expected impact**: +2-5pp overall. May push past 70% without touching features.

**Effort**: An afternoon. No retraining, no new features, no LLM calls. Pure threshold tuning on existing predictions.

**Implementation**:
```python
# Pseudocode
thresholds = {"confident": 0.7, "disputed": 0.5, "qualified": 0.4, "abstain": 0.5}
probas = model.predict_proba(X_test)
for i, proba_row in enumerate(probas):
    max_class = classes[proba_row.argmax()]
    max_proba = proba_row.max()
    if max_proba < thresholds[max_class]:
        prediction = governor_decision[i]  # fallback
    else:
        prediction = max_class
```

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

| Step | Action | Target | Expected Impact | Effort | Retraining? |
|------|--------|--------|----------------|--------|-------------|
| **1** | **Per-class calibrated thresholds** | All | +2-5pp overall | **Low** | No |
| **2** | **Continuous CA signals** | Disputed | +3-5pp disputed | **Low-Medium** | Yes |
| **3** | **Source agreement features** | Confident | +3-5pp confident | **Medium** | Yes |
| **4** | **Governor-classifier hybrid** | All | Architectural | **Medium** | No |
| 5 | Pairwise embedding similarity | Disputed/Confident | +1-2pp | Medium | Yes |
| 6 | Cross-validation + per-class optimization | All | +2-3pp | Low | Yes |
| 7 | Real-world failure collection | All | Compound | Ongoing | Yes |
| 8 | Feature selection / RFE | All | +1-2pp | Medium | Yes |

**Steps 1-4 are the critical path.** Step 1 alone might push past 70% overall. Steps 2-3 target the weakest classes directly. Step 4 makes the system production-ready and auditable.
