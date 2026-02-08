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

## 1. Feature Engineering (Highest Impact)

### Problem: Context length features dominate

The top 3 features by importance are `ctx_length_mean`, `ctx_total_chars`, `ctx_length_std`. These are proxy signals — the model is learning "short contexts = abstain, long contexts = qualified" instead of real governance logic.

### Actions

**A. Improve constraint signal quality**
- The `ca_signal` and `has_disputed_signal` features are the most governance-relevant signals, but they rank #5-6. Tighter CA prompts could make them more discriminative.
- Experiment: Run CA with multiple temperature settings, pick the most calibrated.
- Experiment: Add a second-pass CA specifically for implicit contradictions (the biggest disputed miss pattern).

**B. Add pairwise embedding similarity features**
- Current: TF-IDF pairwise similarity between contexts (computed from raw text).
- Better: Cosine similarity between context embeddings (the actual vectors used in retrieval).
- This directly measures "how similar do the vector DB think these chunks are?" — a stronger signal than TF-IDF.

**C. Add query-context embedding similarity**
- Current: `mean_vector_score` (from eval pipeline).
- Add: Per-chunk vector score distribution features — skewness, kurtosis, percentiles.
- Rationale: Abstain cases should have uniformly low scores. Disputed should have high scores with high variance (relevant but conflicting).

**D. Extract LLM confidence signals**
- The constraints already call LLMs. Extract the LLM's own uncertainty markers from responses — hedging language, "however" in CA verdicts, jury vote margins in AV.
- This is essentially meta-learning: using the LLM's behavior as a feature.

---

## 2. Confident Recall (62% → 70%+)

### Why confident is hard

Confident is the "nothing went wrong" class. The model must learn:
- "Despite surface-level textual variance, sources agree" (different_framing pattern)
- "Despite apparent contradiction, it's resolved" (contradiction_resolved pattern)
- "Despite CA firing, the answer is clear" (opposing_with_consensus pattern)

All three require the model to look past noise. The constraint features see the noise; the model must learn to discount it.

### Actions

**A. Add a "resolved contradiction" feature**
- Binary feature: did CA fire AND do the contexts have a clear temporal ordering (dates) or authority ordering (meta-analysis vs single study)?
- This directly encodes the contradiction_resolved pattern.

**B. Source agreement feature**
- Count how many distinct conclusions the contexts reach. If 3 contexts all conclude the same thing (even in different words), that's a strong confident signal.
- Implementation: LLM call per case — "Do these 3 passages reach the same conclusion? yes/no"
- Cost: 1 additional LLM call per case during feature extraction.

**C. Negative examples in training**
- The model over-hedges because qualified is 2x the size of confident. Even after rebalancing (423 vs 257), qualified dominates.
- Generate more "confident cases that look qualified" — specifically cases where CA fires but the answer is clear.

---

## 3. Qualified Recall (66% → 70%+)

### Why qualified is hard

Qualified is the catch-all with 18 subcategories. It sits at every class boundary:
- Not-quite-abstain (partial evidence)
- Not-quite-disputed (tension but not contradiction)
- Not-quite-confident (evidence exists but needs caveats)

The model confuses it with confident (over-confidence) and disputed (over-sensitivity).

### Actions

**A. Subcategory-aware training**
- Currently all qualified cases are treated equally. But "partial_evidence" qualified is very different from "causal_uncertainty" qualified.
- Experiment: Add subcategory as a feature (label-encoded). This gives the model a hint about the type of qualification needed.
- Risk: Might overfit to subcategory distribution.

**B. Hedging language features**
- Qualified answers typically come from contexts with hedging language: "may", "could", "suggests", "preliminary".
- Add a hedging word count feature (cheap, no LLM).
- Add a hedging ratio: hedging words / total words in contexts.

**C. Evidence completeness feature**
- Qualified often means "the evidence exists but is incomplete."
- Feature: Does the query ask for something specific (a number, a date, a name) that the contexts don't contain?
- Implementation: SIT constraint already detects `info_type_requested` and `has_specific_info`. But this only fires for specific info types. Generalize it.

---

## 4. Disputed Recall (67% → 70%+)

### Why disputed is close but not there

The remaining misses are "quiet disputes" — contradictions without loud markers. The 5 hardest patterns from the test plan:
- `implicit_contradiction`: Contradictions implied by context, not stated
- `binary_conflict`: Clear yes/no disagreement, but model doesn't see it
- `statistical_direction_conflict`: Stats point different directions
- `temporal_conflict`: Time-based conflict confused with "old data"
- `numerical_conflict`: Numbers differ meaningfully but model doesn't catch it

### Actions

**A. Numerical conflict detector improvement**
- Current: `ca_numerical_variance_detected` catches numbers within 25%.
- Problem: Numbers >25% apart that represent the same measurement ARE disputes, but this feature doesn't help distinguish dispute from qualified.
- Add: A "numerical magnitude" feature — how far apart are the most divergent numbers in the contexts?

**B. Temporal conflict detector**
- Current: `detection_temporal` flags temporal queries, but doesn't flag temporal conflicts between sources.
- Add: A feature that detects if contexts reference different time periods for the same claim.
- Implementation: Regex for year patterns + check if multiple years appear in different contexts.

**C. Assertion polarity feature**
- For binary conflicts: extract the polarity of each context's main claim (positive/negative/neutral).
- If one context says "X is safe" and another says "X is not safe", the polarity differs.
- Implementation: Simple negation detection per context + comparison.

---

## 5. Training Improvements

### A. Cross-validation instead of single split
- Currently: Single 80/20 split, seed=42. Results depend on which cases land in test.
- Better: Report mean +/- std across 5 stratified folds. More reliable measurement.
- The 83% → 67% "regression" was partly due to different test splits.

### B. Per-class optimization
- Currently: Model optimizes overall accuracy. This favors qualified (largest class).
- Better: Optimize for minimum per-class recall (ensure no class drops below threshold).
- Implementation: Custom scoring function in RandomizedSearchCV.

### C. Calibrated confidence thresholds
- GBT outputs class probabilities. When max_proba < 0.5, the model is unsure.
- Use probability calibration (Platt scaling or isotonic regression) to get reliable confidence scores.
- Then: Set per-class confidence thresholds. If confident_proba < 0.6, fall back to qualified.
- This directly addresses over-confidence and under-hedging.

### D. Feature selection
- 58 features may include noise. Run recursive feature elimination (RFE) to find the minimal effective set.
- Hypothesis: Removing proxy features (ctx_length_*) might force the model to learn from governance signals.

---

## 6. Data Quality

### A. Audit existing cases
- Some subcategories have <10 cases. These contribute noise, not signal.
- Audit: `code_abstention` (3 cases), `cross_domain_insufficient` (3→10 after expansion), `source_conflict` (4 cases).

### B. Real-world failure collection
- Once integrated into the pipeline, collect cases where the classifier disagrees with human judgment.
- These are the highest-value training examples — real failures, not synthetic ones.

### C. Active learning
- After integration, log all predictions with their probabilities.
- Cases where max_proba is near the decision boundary (0.3-0.5) are the most informative to label next.
- This focuses labeling effort on exactly the cases the model struggles with.

---

## Priority Order

| Priority | Action | Expected Impact | Effort |
|----------|--------|----------------|--------|
| 1 | Cross-validation + per-class optimization | +2-3pp per class | Low |
| 2 | Calibrated confidence thresholds | +3-5pp confident/qualified | Low |
| 3 | Hedging language features | +2-3pp qualified | Low |
| 4 | Numerical magnitude feature | +2-3pp disputed | Low |
| 5 | Pairwise embedding similarity | +1-2pp disputed/confident | Medium |
| 6 | Source agreement feature (LLM) | +3-5pp confident | Medium |
| 7 | Real-world failure collection | Compound | Ongoing |
| 8 | Feature selection / RFE | +1-2pp overall | Medium |

Priorities 1-4 are low-effort, high-impact changes that could push all classes above 70%. Start there.
