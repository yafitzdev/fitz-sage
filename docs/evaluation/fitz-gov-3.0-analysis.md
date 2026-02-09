# fitz-gov 3.0: Technical Analysis & Classifier Investigation

**Purpose**: Deep technical analysis of v3.0 classifier behavior, failure modes, and improvement paths
**Related**: [fitz-gov-3.0-results.md](fitz-gov-3.0-results.md) for results summary

---

## The Shift: Governor Rules to ML Classifier

### Why Rules Failed

The v2.0 governor used priority rules:
```
IE abstain signal    → ABSTAIN   (highest priority)
dispute signal       → DISPUTED
any denial           → QUALIFIED
nothing fired        → CONFIDENT (lowest priority)
```

This achieved 72.3% on 249 cases but had fundamental problems:
1. **Can't handle signal interactions** — dispute signal + high relevance + hedging language might mean "qualified", but rules always pick "disputed"
2. **Over-predicts after tuning** — tighter CA prompts (Exp 3) improved dispute detection but made the governor predict "disputed" for 60% of cases
3. **No learning** — rules can't improve from labeled data

The classifier sees the same constraint outputs as *features* and learns decision boundaries from 1113 labeled examples.

### What Changed Between v2.0 and v3.0

| Aspect | v2.0 (Governor) | v3.0 (Classifier) |
|--------|-----------------|-------------------|
| Decision method | Priority rules | GBT (58 features) |
| Test cases | 249 governance | 1113 governance |
| Evaluation | Full pipeline (LLM) | Feature extraction + predict |
| LLM dependency | Per-query (3b model) | Training only (offline) |
| Inference cost | ~6 LLM calls/query | Microseconds (tree traversal) |
| Variance | High (56-79% qualification) | Low (deterministic) |

---

## Failure Analysis: Why Each Class Struggles

### Abstain (85% recall) — The Easy One

**Why it works**: Abstain has clear trigger signals. When IE fires with "abstain" and entity matching fails, the classifier has strong features:
- `ie_signal = "abstain"` fires on off-topic contexts
- `ie_entity_match_found = False` catches wrong-entity cases
- Short contexts (low `ctx_total_chars`) correlate with insufficient evidence

**Remaining failures (7/47)**: All are abstain→qualified. These are near-miss cases where context is topically related but doesn't answer the specific question. The classifier sees high entity overlap and moderate vector scores, making it hedge instead of abstain.

**Example failure pattern**: Query about "2024 World Series", context discusses "World Series history" generally. Entity "World Series" matches, vector scores are high, but the specific year isn't covered.

### Disputed (67% recall) — Close but Missing Subtle Conflicts

**Why it partially works**: The CA constraint fires on explicit contradictions, giving strong `has_disputed_signal` and `ca_signal` features.

**Why it misses 13/39**: The failures cluster in 5 patterns:

| Pattern | Count | Description |
|---------|-------|-------------|
| Implicit contradiction | 4 | Contradictions implied by context, not stated directly |
| Binary conflict | 3 | Yes/no disagreement that CA doesn't catch |
| Statistical direction | 2 | Stats point different directions but aren't flagged |
| Temporal conflict | 2 | Time-based conflict confused with "old data" |
| Disputed → qualified | 8 | CA fires weakly, classifier hedges instead of flagging |

**The D→Q confusion** (8 cases) is the biggest issue. These are cases where CA fires but other features (moderate vector scores, similar context lengths) push the classifier toward "qualified." The classifier learned that CA firing alone isn't sufficient — which is usually correct, but not always.

**Root cause**: The `has_disputed_signal` feature is binary (0/1). It doesn't capture CA *confidence* — a marginal contradiction gets the same signal as a clear factual disagreement. Adding CA confidence as a continuous feature would help.

### Qualified (66% recall) — The Catch-All Problem

**Why it's hard**: Qualified sits at every class boundary:
- Not-quite-abstain (partial evidence) — 7 cases go to abstain
- Not-quite-disputed (tension without contradiction) — 5 cases go to disputed
- Not-quite-confident (evidence exists but needs caveats) — 8 cases go to confident

It has 18 subcategories spanning fundamentally different epistemic situations:

| Subcategory cluster | Examples | What makes it qualified |
|---------------------|----------|----------------------|
| Partial evidence | partial_evidence, incomplete_data | Has something but not enough |
| Causal uncertainty | causal_uncertainty, correlation_not_causation | Asks "why" but evidence is correlational |
| Scope limits | geographic_limits, temporal_limits | Answer is true but context-dependent |
| Ambiguity | entity_ambiguity, metric_ambiguity | Multiple interpretations |
| Missing specifics | missing_quantification, approximate_data | Has the answer type but not the exact value |

**Root cause**: No single feature distinguishes qualified from its neighbors. The classifier relies on proxy signals (context length, pairwise similarity) rather than semantic understanding of "this evidence needs caveats."

**The 8 qualified→confident failures** are the most dangerous error mode — the system answers without hedging when it should. These tend to be cases with high vector scores and no constraint firing, where the classifier can't see that the evidence is incomplete.

### Confident (62% recall) — Learning the Absence of Problems

**Why it's hardest**: Confident is the "nothing went wrong" class. The classifier must learn:
1. "Despite surface-level textual variance, sources agree" — different_framing pattern
2. "Despite apparent contradiction, it's resolved" — contradiction_resolved pattern
3. "Despite CA firing, the answer is clear" — opposing_with_consensus pattern

All three require looking past noise. The constraint features see the noise (CA fires, hedging language detected); the classifier must learn to discount it.

**15 confident→qualified failures**: The classifier over-hedges. When any constraint fires even mildly, or when context features suggest complexity (high ctx_length_std, multiple numbers), it defaults to "qualified" — the safest prediction.

**3 confident→disputed failures**: CA fires on apparent contradictions that are actually resolved (one source supersedes the other). The classifier doesn't have a "resolved contradiction" feature — it just sees `has_disputed_signal = True` and sometimes follows it.

**Root cause**: The classifier needs *positive* signals for confidence (sources agree, evidence is complete, no caveats needed), not just the absence of negative signals. Currently, confident is predicted when nothing else scores high enough — it's the default, not a deliberate detection.

---

## Feature Analysis: What the Classifier Actually Learned

### Context Features Dominate (Proxy Problem)

The top 3 features (ctx_length_mean, ctx_total_chars, ctx_length_std) account for 28.4% of total importance. These are proxies:
- Short contexts → abstain (insufficient evidence)
- Long, similar-length contexts → confident (multiple sources agree)
- Long, varied-length contexts → qualified or disputed (asymmetric evidence)

This works because fitz-gov's synthetic contexts correlate length with mode — abstain cases intentionally have short, irrelevant contexts. But in production, context length depends on the document corpus, not the governance mode.

**Risk**: The classifier may be overfit to fitz-gov's context length distribution. Real-world contexts will have different length profiles.

### Vector Score Features Are Discriminative

`mean_vector_score` (#4, 6.3%) became important in Exp 5 when real embeddings were added. The distribution by class:

| Class | mean_vector_score (approx) | Interpretation |
|-------|---------------------------|----------------|
| Abstain | Low (0.3-0.5) | Contexts aren't relevant to query |
| Disputed | High (0.7-0.9) | Contexts are relevant but contradict |
| Qualified | Medium (0.5-0.7) | Contexts are partially relevant |
| Confident | High (0.7-0.9) | Contexts are relevant and agree |

Disputed and confident both have high vector scores — distinguishing them requires the CA constraint signal on top of the score.

### Constraint Signals Are Underweight

`has_disputed_signal`, `ca_signal`, and `ca_fired` together account for 15.2% — less than context length alone (28.4%). This inversion means the classifier trusts proxy signals more than governance-specific signals.

**Why**: Constraint signals are sparse (CA only fires on ~30% of cases) and binary. Context features are continuous and always present. Trees naturally favor features with more split opportunities.

**Fix path**: Make constraint signals richer — add CA confidence scores, evidence character breakdown, pairwise agreement counts. See `CLASSIFIER_NEXT_STEPS.md`.

---

## The Disputed Recall Regression (RF 83% → 72%, GBT 67%)

### What Happened

| Experiment | Dataset | Model | Disputed Recall | Test Cases |
|------------|---------|-------|-----------------|------------|
| Exp 5 | 914 cases | RF (tuned) | **83%** (24/29) | 29 disputed |
| Exp 6 | 1113 cases | GBT (tuned) | **67%** (26/39) | 39 disputed |

### Investigation Findings

1. **The real regression is smaller than reported**: RF on 1113 data gets 72% disputed (not 67%). The 83%→67% comparison mixed model types (RF→GBT).

2. **New cases are harder by design**: The 51 new disputed cases target implicit contradictions, binary conflicts, and temporal conflicts — patterns that were specifically chosen because the classifier missed them.

3. **Test set composition changed**: 29→39 disputed test cases. The 10 additional test cases include harder patterns, lowering the average.

4. **GBT vs RF tradeoff**: RF gets better disputed recall (72%) but worse qualified recall (55%). GBT is more balanced (67%/66%). The user chose GBT because 55% qualified recall is unacceptable.

5. **Hyperparameter tuning is critical**: Untuned RF gets 28% disputed recall. The initial regression analysis used untuned models and reported a misleading 31% figure.

### Statistical Reliability

| Test set size | 95% CI for 67% recall |
|---------------|----------------------|
| 29 cases (Exp 5) | 48% - 83% |
| 39 cases (Exp 6) | 50% - 81% |

With 39 test cases, the 95% confidence interval is still wide (50-81%). The difference between 67% and 83% could be noise. Reaching 100 disputed test cases would narrow the CI to ±9pp.

---

## Optimization Attempts That Failed

### Experiment 7a: New Text Features

Added 6 features: `ctx_hedging_count`, `ctx_assertive_count`, `ctx_hedging_ratio`, `ctx_assertive_ratio`, `ctx_unique_number_count`, `ctx_exclusive_numbers_ratio`.

**Result**: GBT dropped from 69.1% to 66.8%. The features correlated with context length (already #1) and added noise.

**Lesson**: New features must provide *orthogonal* signal. Hedging counts scale with text length — the classifier already knows text length.

### Experiment 7b: Extended Hyperparameter Search

Tripled search budget from 200s to 600s per model.

**Result**: GBT found max_depth=2 (vs 6), scoring 60.1%. The longer search explored a different region of hyperparameter space and converged on a shallow model that can't capture feature interactions.

**Lesson**: RandomizedSearchCV is stochastic. More iterations don't guarantee better results — they can find different (worse) local optima.

---

## Production Integration Considerations

### Governor Fallback Strategy

The classifier outputs class probabilities. When `max_proba` is low, the prediction is uncertain. Planned approach:

```
if classifier.max_proba >= threshold:
    use classifier prediction
else:
    use governor priority-rule prediction
```

The governor already runs (constraints are the classifier's feature extractors), so this adds zero latency. The threshold needs calibration — Platt scaling or isotonic regression on a held-out set.

### Feature Availability at Runtime

All 58 features are available at inference time:
- **Tier 1** (constraint metadata): Always available — constraints run before the decision
- **Tier 2** (vector scores): Available from VectorSearchStep results
- **Tier 3** (detection flags): Available from DetectionOrchestrator (already runs during retrieval)
- **Context features**: Computed from raw chunk text (no LLM needed)

### Model Size and Latency

- Model artifact: ~5MB (joblib serialized GBT)
- Inference: <1ms (tree traversal on 58 features)
- Memory: negligible (loaded once at engine startup)

---

## Path to 70%+ Per-Class Recall

See `CLASSIFIER_NEXT_STEPS.md` for the full roadmap. Summary of highest-impact actions:

| Priority | Action | Target Class | Expected Impact |
|----------|--------|-------------|-----------------|
| 1 | Cross-validation + per-class optimization | All | +2-3pp per class |
| 2 | Governor fallback for low confidence | Confident, Qualified | +3-5pp |
| 3 | Richer CA signals (confidence, not just binary) | Disputed | +2-3pp |
| 4 | Numerical magnitude feature | Disputed | +2-3pp |
| 5 | Source agreement LLM feature | Confident | +3-5pp |
| 6 | Real-world failure collection | All | Compound |

The ceiling with current features/data is 69.1%. Breaking through requires structural improvements to the feature extraction layer.
