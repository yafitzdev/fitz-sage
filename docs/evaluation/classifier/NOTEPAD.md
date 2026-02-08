# Governance Classifier — Living Notepad

**Goal**: Replace hand-coded `AnswerGovernor.decide()` priority rules with a trained tabular classifier.
**Status**: Training pipeline complete. Best model: RF at 71.0% (vs 33.3% governor baseline on synthetic data). Dispute recall improved from 28% to 69% via ensemble.

---

## 1. Problem Statement

The current governance system uses **staged constraints → priority rules**:

```
IE abstain signal    → ABSTAIN   (highest priority)
dispute signal       → DISPUTED
any denial           → QUALIFIED
nothing fired        → CONFIDENT (lowest priority)
```

This breaks on boundary cases where multiple signals compete. Example: "Evolving facts with source quality asymmetry" triggers both dispute AND qualify — the priority rule picks dispute, but the correct answer is qualify. The classifier learns signal *interactions* instead of hard-coded priority.

### What the classifier does

```
Pipeline runs constraints as usual (feature extractors)
        ↓
Collect ~25 numeric/categorical features
        ↓
Tiny tabular classifier (gradient-boosted tree, KB-sized)
        ↓
Output: one of 4 governance modes (abstain/disputed/qualified/confident)
```

Constraints still run. They become feature extractors instead of decision-makers.

---

## 2. The 4 Governance Modes (No 5th Needed)

| Mode | When | Pipeline signal today |
|------|------|----------------------|
| **ABSTAIN** | Context doesn't answer the query | IE fires with "abstain" |
| **DISPUTED** | Sources make mutually exclusive claims | ConflictAware fires |
| **QUALIFIED** | Answer needs caveats/hedging | Any constraint denies |
| **CONFIDENT** | Clear, consistent evidence | Nothing fires |

### Why no "sufficiency" mode

The sufficiency gap (topically related context, missing specific answer) maps to **qualified**, not a new mode. The taxonomy already covers this in the Abstain↔Qualify boundary:
- "Related topic, missing specific info" → qualified
- "Right topic, wrong info type" → qualified
- "Partial answer" → qualified

The problem was misrouting to confident, not a missing category. The classifier will learn this from labeled boundary cases.

---

## 3. Training Data: fitz-gov

**Repository**: `C:\Users\yanfi\PycharmProjects\fitz-gov`

**Data structure**:
```
fitz-gov/data/
├── tier0_sanity/    60 cases (44 governance, 16 grounding/relevance)
├── tier1_core/      914 cases (848 governance, 66 grounding/relevance)
├── corpus/          378 test documents
└── queries/         Query-to-document mappings
```

### Usable cases: 848 tier1 + 44 tier0 = 892 total

| Category | Tier 0 | Tier 1 | Total | `expected_mode` | Usable? |
|----------|--------|--------|-------|-----------------|---------|
| Abstention | 12 | 192 | 204 | abstain | Yes |
| Confidence | 10 | 154 | 164 | confident | Yes |
| Dispute | 12 | 145 | 157 | disputed | Yes |
| Qualification | 10 | 357 | 367 | qualified | Yes |
| Grounding | 8 | 34 | 42 | qualified* | No — tests answer quality, not mode |
| Relevance | 8 | 32 | 40 | qualified* | No — tests answer content, not mode |

*Grounding/relevance cases evaluate the generated answer text (regex + forbidden claims), not the governance mode decision. Including them would add noise.

**Data expansion history**:
- v1.0: 200 initial cases from 21 experiments (hand-crafted, easy/medium difficulty). Split into tier0 (60) and tier1 (141).
- v2.0: +525 generated via LLM-assisted boundary sampling across 7 batches:
  - Pure abstain+dispute (90), pure qualify+confident (95)
  - D-Q boundary (140 across 2 batches — primary bottleneck)
  - Abstain boundary (65), confident boundary (45)
  - Three-way ambiguity (90)
  - Mode distribution: 102 abstain, 70 disputed, 261 qualified, 92 confident
  - Validated at 95.4% agreement, 5 dupes removed, 7 relabeled, 513 merged into tier1
- v3.0: +123 generated (dispute boundary, edge cases, code/adversarial), validated at 94% agreement, 4 relabeled, all merged

### Class distribution (848 tier1 cases)

```
qualified:  357 (42.1%)
abstain:    192 (22.6%)
confident:  154 (18.2%)
disputed:   145 (17.1%)
```

Imbalance is moderate (2.5x max/min). Workable without oversampling. Disputed class significantly strengthened from 109 to 145 (33% increase).

### Difficulty distribution

| Tier | Easy | Medium | Hard | Total |
|------|------|--------|------|-------|
| Tier 0 (sanity) | 44 | 0 | 0 | 44 |
| Tier 1 (core) | 0 | ~50 | ~675 | 725 |

Heavily weighted toward hard cases (92% of Tier 1).

### Case structure

Each case provides:
- `query`: The question text
- `contexts`: 1-4 context passages
- `expected_mode`: Ground truth label
- `subcategory`: Canonical case type (54 consolidated subcategories)
- `original_subcategory`: Pre-consolidation slug (521 cases preserve this)
- `difficulty`: easy/medium/hard
- `rationale`: Why this mode is correct

### Subcategory coverage (54 consolidated, was 156 raw slugs)

| Category | Subcategories | Largest | Smallest |
|----------|---------------|---------|----------|
| Abstention | 14 | wrong_entity (21) | code_abstention (3) |
| Confidence | 11 | clear_explanation (21) | conditional_confidence (6) |
| Dispute | 11 | numerical_conflict (21) | source_conflict (4) |
| Qualification | 18 | different_aspects (35) | implicit_assumptions (7) |

Consolidation script: `fitz-gov/scripts/consolidate_subcategories.py`
Original slugs preserved in `original_subcategory` field for traceability.
Only 3 subcategories have <5 cases (code_abstention:3, cross_domain_insufficient:3, source_conflict:4).

### Boundary cases (highest-value for classifier)

| Boundary pair | Case types | Why it matters |
|---------------|-----------|----------------|
| Dispute ↔ Qualify | 14 types | Primary bottleneck (34 failures in experiments) |
| Abstain ↔ Qualify | 5 types | Sufficiency gap lives here |
| Abstain ↔ Confident | 6 types | Decoy/adjacent entity confusion |
| Qualify ↔ Confident | 6 types | Over-hedging vs appropriate confidence |
| Dispute ↔ Confident | 3 types | Apparent contradiction that resolves |
| Abstain ↔ Dispute | 2 types | Off-topic contradictions |
| Three-way ambiguity | 13 types | Where priority rules fundamentally break |
| Four-way ambiguity | 2 types | Hardest possible cases |

---

## 4. Available Features (~25 signals)

All signals are already computed by the constraint pipeline before `AnswerGovernor.decide()` runs.

### 4a. Retrieval signals (numeric)

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `max_similarity` | float [0-1] | IE constraint | Highest cosine similarity query↔chunk |
| `max_vector_score` | float [0-1] | VectorDB | Highest dense retrieval score |
| `max_rerank_score` | float [0-1] | RerankStep | Cross-encoder relevance (if enabled) |
| `chunk_count` | int | Pipeline | Number of chunks retrieved |

### 4b. InsufficientEvidence signals

| Feature | Type | Source file | Description |
|---------|------|-------------|-------------|
| `entity_match_found` | bool | insufficient_evidence.py:817 | Query entities found in chunks |
| `primary_match_found` | bool | insufficient_evidence.py:821 | Main subject present in context |
| `critical_match_found` | bool | insufficient_evidence.py:825 | Years, qualifiers all match |
| `aspect_mismatch` | bool | insufficient_evidence.py:882 | Query asks CAUSE, chunks have SYMPTOM |
| `summary_overlap` | bool | insufficient_evidence.py:855 | Query topics in chunk summaries |
| `ie_signal` | cat | insufficient_evidence.py | "abstain" or null |

### 4c. ConflictAware signals

| Feature | Type | Source file | Description |
|---------|------|-------------|-------------|
| `contradiction_detected` | bool | conflict_aware.py:350 | LLM found opposing claims |
| `evidence_character` | cat | conflict_aware.py:200 | "assertive" / "hedged" / "mixed" |
| `numerical_variance` | bool | conflict_aware.py:244 | Same direction, ≤25% difference |
| `ca_signal` | cat | conflict_aware.py | "disputed" or null |

### 4d. CausalAttribution signals

| Feature | Type | Source file | Description |
|---------|------|-------------|-------------|
| `query_type` | cat | causal_attribution.py:204 | "causal"/"predictive"/"opinion"/"speculative"/"none" |
| `has_causal_evidence` | bool | causal_attribution.py:221 | Chunks contain causal language |
| `has_predictive_evidence` | bool | causal_attribution.py:236 | Chunks contain forecasting language |

### 4e. SpecificInfoType signals

| Feature | Type | Source file | Description |
|---------|------|-------------|-------------|
| `info_type_requested` | cat | specific_info_type.py:139 | "pricing"/"quantity"/"temporal"/etc. or null |
| `has_specific_info` | bool | specific_info_type.py:241 | Chunks contain the requested info type |
| `sit_entity_mismatch` | bool | specific_info_type.py:92 | Query entity ≠ chunk entity |

### 4f. AnswerVerification signals

| Feature | Type | Source file | Description |
|---------|------|-------------|-------------|
| `jury_votes_no` | int [0-3] | answer_verification.py:100 | Count of "insufficient" votes from 3-prompt jury |

### 4g. DetectionOrchestrator signals

| Feature | Type | Source | Description |
|---------|------|--------|-------------|
| `has_temporal_intent` | bool | detection/modules/temporal.py | Time-based query |
| `has_aggregation_intent` | bool | detection/modules/aggregation.py | List/count query |
| `has_comparison_intent` | bool | detection/modules/comparison.py | A-vs-B query |
| `boost_recency` | bool | detection/modules/freshness.py | Needs recent sources |
| `boost_authority` | bool | detection/modules/freshness.py | Needs authoritative sources |

### Signal summary

| Category | Count | LLM required? |
|----------|-------|---------------|
| Retrieval scores | 4 | No |
| IE relevance | 6 | Optional (entity extraction LLM fallback) |
| Conflict detection | 4 | Yes (pairwise comparison) |
| Causal attribution | 3 | No (keyword-based) |
| Specific info type | 3 | No (regex-based) |
| Answer verification | 1 | Yes (3-prompt jury) |
| Query classification | 5 | Yes (detection orchestrator) |
| **Total** | **~26** | |

---

## 5. Architecture Decision

**Replace `AnswerGovernor.decide()` with classifier. Keep constraints as feature extractors.**

Current flow:
```
Query → Constraints run → Priority rules → GovernanceDecision
```

New flow:
```
Query → Constraints run → Feature vector → Classifier → GovernanceDecision
```

The constraints still run identically. Only the decision logic changes. If the classifier fails or regresses, fallback to the old priority rules is a one-line swap.

### Why gradient-boosted trees

- 892 examples, ~25 features → textbook tabular classification
- No neural networks, no GPU, no deep learning
- scikit-learn or XGBoost — training is 3-5 lines of code
- Model size: KB-range, microsecond inference
- Handles mixed numeric/categorical features natively
- Built-in feature importance (explains decisions)
- Cross-validation works well at this scale

---

## 6. Due Diligence

### 6a. Test Case Gaps — RESOLVED

All major gaps have been closed. Summary of actions taken:

#### Completed actions

| Action | Status | Details |
|--------|--------|---------|
| Merge staging v1 (525 cases) | DONE | Merged in previous session |
| Consolidate subcategory slugs (156 -> 54) | DONE | `scripts/consolidate_subcategories.py` |
| Generate dispute boundary cases | DONE | 48 cases (25 disputed, 13 qualified, 10 abstain) |
| Generate empty-context cases | DONE | 15 cases (all abstain) |
| Generate short-query cases | DONE | 10 cases (all 4 modes) |
| Generate long-context cases | DONE | 15 cases (all 4 modes) |
| Generate code/structured cases | DONE | 20 cases (all 4 modes) |
| Generate multi-entity comparison cases | DONE | 5 cases |
| Generate adversarial/trick query cases | DONE | 10 cases (false premises, leading, negation) |
| Blind validate all new cases | DONE | 94% agreement, 4 relabeled |
| Merge staging v2 (123 cases) | DONE | `scripts/merge_staging_v2.py` |

#### Final class distribution (848 tier1 cases)

| Class | Count | % | 80/20 test set | Per-misclass impact |
|-------|-------|---|----------------|---------------------|
| qualified | 357 | 42.1% | ~71 | 1.4% |
| abstain | 192 | 22.6% | ~38 | 2.6% |
| confident | 154 | 18.2% | ~31 | 3.2% |
| disputed | 145 | 17.1% | ~29 | 3.4% |

Disputed class now has 29 test cases in 80/20 split (was 21 before expansion). Statistically sufficient for detecting >5% accuracy differences.

#### Remaining minor gaps (acceptable for initial training)

| Gap | Status | Notes |
|-----|--------|-------|
| Multi-language | Not addressed | Classifier is language-agnostic (numeric features only) |
| Medium-difficulty cases | Few | 93% hard — aspirational but not diagnostic |
| Singleton subcategories | 3 remain (<5 cases) | code_abstention:3, cross_domain_insufficient:3, source_conflict:4 |

---

### 6b. Feature Extraction Gaps

The initial inventory listed ~26 features. Deep audit found **~50 usable features** across 4 cost tiers.

#### Tier 1: Already computed inside constraints, just not surfaced (FREE)

These values are calculated during constraint execution but only used internally. Surfacing them requires adding keys to `ConstraintResult` metadata — no new computation.

| Feature | Type | Source | Value for classifier |
|---------|------|--------|---------------------|
| `ie_entity_match_found` | bool | IE `_check_embedding_relevance` | **High** — entity presence is a core relevance signal |
| `ie_primary_match_found` | bool | IE internal | **High** — main subject detection |
| `ie_critical_match_found` | bool | IE internal | **High** — years, qualifiers all match |
| `ie_query_aspect` | cat (12 values) | AspectClassifier | **High** — CAUSE/SYMPTOM/TREATMENT/etc., uniquely valuable |
| `ie_has_matching_aspect` | bool | AspectClassifier | Medium — aspect compatibility |
| `ie_has_conflicting_aspect` | bool | AspectClassifier | Medium — aspect incompatibility |
| `ie_summary_overlap` | bool | IE `_has_summary_overlap` | Medium |
| `ie_entity_overlap` | bool | IE `_has_entity_overlap` | Medium |
| `conflict_numerical_variance` | bool | NumericalConflictDetector | **High** — strong negative predictor for DISPUTED |
| `conflict_is_uncertainty_query` | bool | CA `_is_uncertainty_query` | Medium |
| `conflict_skipped_hedged_pairs` | int | CA internal | **High** — how many pairs skipped due to hedging |
| `ca_has_causal_evidence` | bool | CA `_has_appropriate_evidence` | Medium — only surfaced when denied, not when allowed |
| `ca_has_predictive_evidence` | bool | CA internal | Medium |
| `ca_is_forecast_query` | bool | CA `_mentions_future_year` | Medium |
| `av_vote_distribution` | int | AV jury internals | Medium — complement of jury_votes_no |

**Implementation**: Add these keys to the `metadata` dict in each constraint's `deny()` and `allow()` calls. ~2 hours of work.

#### Tier 2: Cheap new computation on existing data (NO LLM, NO I/O)

Trivial calculations on data already available at constraint time.

| Feature | Type | Computation | Value for classifier |
|---------|------|-------------|---------------------|
| `num_chunks` | int | `len(chunks)` | **High** |
| `num_unique_sources` | int | `len(set(c.doc_id for c in chunks))` | **High** — source diversity, uniquely valuable |
| `mean_vector_score` | float | Mean of chunk vector_scores | **High** |
| `std_vector_score` | float | Std dev of vector_scores | **High** — score distribution shape |
| `score_spread` | float | max - min vector_score | **High** — uniquely valuable for confident vs qualified |
| `query_word_count` | int | `len(query.split())` | Medium |
| `vocab_overlap_ratio` | float | `len(query_words & context_words) / len(query_words)` | Medium |
| `dominant_content_type` | cat | Most common content_type from chunk enrichment | Medium |
| `from_sparse_count` | int | Chunks from BM25 | Medium — how indirect was retrieval |
| `from_entity_graph_count` | int | Chunks from entity graph | Low-Medium |

**Implementation**: Build a `FeatureExtractor` class that takes `query + chunks + constraint_results` and outputs a feature dict. ~3 hours.

#### Tier 3: Thread DetectionSummary to constraints (MEDIUM effort)

The DetectionOrchestrator already computes these during retrieval, but the data is lost before it reaches the constraint pipeline.

| Feature | Type | Value for classifier |
|---------|------|---------------------|
| `detection_temporal` | bool | **High** — temporal queries have unique governance patterns |
| `detection_aggregation` | bool | **High** — list/count queries behave differently |
| `detection_comparison` | bool | **High** — need both entities present, uniquely valuable |
| `detection_freshness_recency` | bool | Medium |
| `detection_freshness_authority` | bool | Medium |
| `detection_needs_rewriting` | bool | Medium |
| `detection_comparison_entity_count` | int | Medium |

**Implementation**: Thread `DetectionSummary` from `VectorSearchStep` through retrieval return to the constraint pipeline. Requires changing the interface between retrieval and constraints. ~4 hours of refactoring.

#### Tier 4: Skip (low value or redundant)

| Feature | Why skip |
|---------|----------|
| `query_char_count` | Correlated with word_count |
| `ie_lexical_overlap` | Subsumed by summary_overlap |
| `has_entity_enrichment` / `has_summary_enrichment` | Almost always True after ingestion |
| `mean_chunk_length` | Low predictive value |
| `has_rrf_scores` | Low predictive value |
| New LLM-based features | Not justified — existing LLM calls already cover these signals |

#### Redundancy analysis

| Pair | Verdict |
|------|---------|
| `max_vector_score` vs `max_similarity` | Same thing at different pipeline stages. **Keep `max_similarity` only** |
| `ca_fired` vs `ca_query_type != "none"` | Perfectly correlated. **Keep `query_type` (richer)** |
| `conflict_evidence_characters` vs `first_char` + `pair_char` | Former is pair encoding. **Keep decomposed for classifier** |
| `ie_entity_overlap` vs `ie_entity_match_found` | Different logic (enriched vs heuristic). **Keep both** |
| `num_constraints_fired` vs individual `*_fired` bools | Count derivable from bools. **Keep both** — count provides aggregate |

#### Features with unique predictive value (not captured elsewhere)

| Feature | Why unique |
|---------|-----------|
| `ie_query_aspect` | Only query intent classification (CAUSE/SYMPTOM/etc.) |
| `conflict_numerical_variance` | Strong negative predictor for DISPUTED — "same direction, close values" |
| `num_unique_sources` | Source diversity — multiple sources agree (confident) vs disagree (disputed) |
| `score_spread` | Retrieval confidence distribution — tight cluster vs one outlier |
| `detection_comparison` | Comparison queries have unique governance needs |

#### Recommended feature count by implementation phase

| Phase | Features | Effort | Description |
|-------|----------|--------|-------------|
| Start training | ~17 | 0h | Features already in ConstraintResult metadata |
| + Tier 1 surfacing | ~32 | 2h | Surface internal constraint values |
| + Tier 2 computation | ~42 | 3h | Cheap text statistics and score distributions |
| + Tier 3 threading | ~49 | 4h | Thread DetectionSummary to constraints |

**Recommendation**: Start with Tier 1+2 (~42 features, ~5h work) for initial training. Add Tier 3 only if feature importance analysis shows query classification gaps.

#### Implementation Status: COMPLETE

All tiers (1-3) are now implemented in the codebase:

| Tier | Status | Files changed |
|------|--------|--------------|
| Tier 1: Constraint internals | DONE | `insufficient_evidence.py`, `conflict_aware.py`, `causal_attribution.py`, `specific_info_type.py`, `base.py` |
| Tier 2: FeatureExtractor class | DONE | `core/guardrails/feature_extractor.py` (new) |
| Tier 3: DetectionSummary threading | DONE | `vector_search.py`, `retrieval/loader.py`, `pipeline/engine.py`, `staged.py` |

**Key changes**:
- `ConstraintResult.allow()` now accepts `**metadata` kwargs (same as `deny()`)
- All constraints surface diagnostic dicts in both allow and deny paths
- `staged.py` always injects `constraint_name` + `stage` into result metadata
- `feature_extractor.py` extracts ~40 features from query + chunks + constraint results + detection summary
- DetectionSummary threaded from VectorSearchStep → retrieval loader → RAGPipeline engine
- Governance features extracted at Step 2.5 in the pipeline (after constraints, before governor)

---

## 7. Training Pipeline

### 7a. Feature Extraction Script

**File**: `tools/governance/extract_features.py`

Loads labeled cases from fitz-gov, runs each constraint individually (bypassing staged short-circuit), extracts features via `feature_extractor.py`, saves to CSV.

```bash
python -m tools.governance.extract_features --chat cohere --workers 1
```

**Key design decisions**:
- **Constraints run individually, not staged** — The staged pipeline short-circuits (if IE fires abstain, CA/AV never run). For training we need ALL features, so each constraint runs independently.
- **Thread-local constraint instances** — Each worker thread gets its own constraint set to avoid shared state.
- **Governor baseline** — Also runs `AnswerGovernor.decide()` on each case's results, stored as `governor_predicted` for comparison.
- **Typed defaults** — None values replaced with False (bool), 0 (numeric), "none" (string).

**LLM requirement**: IE, CA, and AV constraints use LLM calls. ~5-7 LLM calls per case. Cohere command-r7b via fast tier. 914 cases = ~5000 LLM calls.

**Concurrency lessons learned**:
- 100 workers: massive 429 rate limits, AV features almost all zeros (879/914 had 0 jury votes)
- 10 workers: still hit limits
- 3 workers: some 429s mid-run, ~80% clean data
- 2 workers: occasional 429s, ~90% clean data
- **1 worker: zero errors, 100% clean data, ~15 min runtime**

**Output**: `tools/governance/data/features.csv` — 914 rows x 52 columns

**Data note**: fitz-gov has 914 tier1 cases (was 848 at last doc update — the 66 grounding/relevance cases map to `expected_mode=qualified` and are included since the classifier just predicts the mode, not the reason).

### 7b. Training Script

**File**: `tools/governance/train_classifier.py`

```bash
python -m tools.governance.train_classifier --time-budget 600
```

**Pipeline**:
1. Load features.csv (52 constraint-derived columns)
2. Enrich with 11 context-based features from raw case text (no LLM)
3. Encode categoricals (LabelEncoder), bools to int
4. Stratified 80/20 train/test split
5. Quick model comparison (5 models, 5-fold CV, class-weighted)
6. Hyperparameter search on top 3 models (RandomizedSearchCV with time budget)
7. Build stacking ensemble from tuned models
8. Evaluate all models + ensemble on held-out test set
9. Save winner as joblib artifact

**Features**: 58 total (47 constraint-derived + 11 context-based)

---

## 8. Training Experiments

### Experiment 1: Baseline (single GBT, constraint features only)

**Config**: GradientBoostingClassifier, n_estimators=200, max_depth=4, lr=0.1, no class weighting, 47 features.

```
CV accuracy: 0.595 +/- 0.016

                 precision    recall  f1-score   support
     abstain       0.75      0.62      0.68        39
   confident       0.39      0.35      0.37        31
    disputed       0.50      0.28      0.36        29
   qualified       0.58      0.74      0.65        84

Classifier accuracy: 0.574 (105/183)
Governor baseline:   0.333 (61/183)
Delta:               +0.240
```

**Dispute diagnostic**: 8/29 correct (28% recall), 17/29 misclassified as qualified.

**Feature importance** (top 5): query_word_count (0.18), vocab_overlap_ratio (0.17), num_chunks (0.12), av_jury_votes_no (0.08), num_constraints_fired (0.07).

**Key insight**: Generic query features dominate over constraint signals. Classifier is learning surface-level patterns, not governance logic. Dispute recall is terrible because CA only fired on 221/914 cases.

### Experiment 2: All improvements (v2)

**4 improvements applied simultaneously**:
1. **Context features** (+11 features): ctx_length_mean/std/total, ctx_contradiction_count, ctx_negation_count, ctx_number_count/variance, ctx_pairwise_sim (max/mean/min), query_ctx_content_overlap
2. **Class weighting**: `compute_sample_weight("balanced")` for GBT, `class_weight="balanced"` for RF/ET/SVM/LR
3. **Multi-model comparison**: GBT, RandomForest, ExtraTrees, SVM, LogisticRegression
4. **Hyperparameter search**: RandomizedSearchCV, 600s budget, top 3 models, 330 fits each

**Quick model comparison (5-fold CV, class-weighted)**:

| Model | CV Accuracy | Time |
|-------|-------------|------|
| ET    | 0.711 +/- 0.031 | 2.8s |
| RF    | 0.699 +/- 0.018 | 3.5s |
| GBT   | 0.683 +/- 0.028 | 14.3s |
| LR    | 0.424 +/- 0.074 | 0.6s |
| SVM   | 0.250 +/- 0.105 | 0.9s |

SVM and LR were poor (features aren't scaled, and the problem isn't linearly separable).

**Hyperparameter search results** (600s budget):

| Model | Best CV Score | Key Params |
|-------|--------------|------------|
| ET    | 0.707 | n_estimators=502, max_depth=30, max_features=0.7 |
| RF    | 0.702 | n_estimators=502, max_depth=30, max_features=0.7 |
| GBT   | 0.672 | n_estimators=485, max_depth=3, lr=0.016, subsample=0.67 |

**Test set results (held-out 183 samples)**:

| Model | Accuracy | Disputed Recall | D->Q Confusion |
|-------|----------|-----------------|----------------|
| Governor baseline | 33.3% (61/183) | 55% (16/29) | N/A |
| ET (tuned) | 65.6% (120/183) | 34% (10/29) | 14/29 |
| GBT (tuned) | 66.7% (122/183) | **62% (18/29)** | **8/29** |
| **RF (tuned)** | **71.0% (130/183)** | 45% (13/29) | 16/29 |
| Stacking Ensemble | 66.1% (121/183) | **69% (20/29)** | **5/29** |

**Per-class breakdown (RF — best overall)**:

```
              precision    recall  f1-score   support
     abstain       0.94      0.77      0.85        39
   confident       0.64      0.45      0.53        31
    disputed       0.76      0.45      0.57        29
   qualified       0.65      0.87      0.74        84
```

**Per-class breakdown (Ensemble — best dispute detection)**:

```
              precision    recall  f1-score   support
     abstain       0.80      0.82      0.81        39
   confident       0.47      0.45      0.46        31
    disputed       0.54      0.69      0.61        29
   qualified       0.72      0.65      0.69        84
```

**Feature importance (RF, top 20)**:

| Rank | Feature | Importance | New? |
|------|---------|------------|------|
| 1 | ctx_total_chars | 0.1137 | YES |
| 2 | ctx_length_mean | 0.1059 | YES |
| 3 | ctx_length_std | 0.0792 | YES |
| 4 | ca_signal | 0.0607 | |
| 5 | ctx_number_variance | 0.0602 | YES |
| 6 | ctx_number_count | 0.0597 | YES |
| 7 | query_word_count | 0.0548 | |
| 8 | ctx_min_pairwise_sim | 0.0469 | YES |
| 9 | av_jury_votes_no | 0.0412 | |
| 10 | ctx_max_pairwise_sim | 0.0406 | YES |
| 11 | av_fired | 0.0388 | |
| 12 | ctx_mean_pairwise_sim | 0.0387 | YES |
| 13 | query_ctx_content_overlap | 0.0355 | YES |
| 14 | vocab_overlap_ratio | 0.0353 | |
| 15 | ctx_negation_count | 0.0242 | YES |
| 16 | num_chunks | 0.0215 | |
| 17 | ca_fired | 0.0198 | |
| 18 | has_disputed_signal | 0.0184 | |
| 19 | ctx_contradiction_count | 0.0164 | YES |
| 20 | caa_has_causal_evidence | 0.0116 | |

**11 of top 20 features are the new context-based ones.** They provide raw text signal that constraint features miss.

### Key Tradeoff: Accuracy vs Dispute Recall

There's a fundamental tension between overall accuracy and dispute detection:

- **RF (71.0% accuracy)** is best overall but only catches 45% of disputes
- **Ensemble (66.1% accuracy)** catches 69% of disputes but sacrifices 5% overall
- **GBT (66.7% accuracy)** catches 62% of disputes with minimal overall cost

This suggests we might want: RF for the general case, with dispute-specific thresholds or a two-stage approach (RF predicts, then dispute-check on qualified predictions).

### Why Governor Baseline is 33.3% (not ~70%)

The governor gets ~70% in the full eval pipeline (real retrieval, vector scores, detection summaries). Here it gets 33.3% because:

1. **No real retrieval signals** — Synthetic cases have inline contexts with no vector scores, no embedding similarities, no metadata. IE sees `max_similarity=0` → doesn't fire → governor defaults to "confident".
2. **Governor predicted "confident" on 472/914 cases** but only 157 should be — it's just the default when nothing fires.
3. **Governor predicted "abstain" on only 55 cases** but 192 are expected — IE can't detect insufficient evidence without real embeddings.

The classifier compensates by learning from text-level proxies (context length, word overlap, contradiction markers) that the governor's priority rules can't access. With real pipeline data, the classifier would have much richer features and should significantly exceed the governor's ~70%.

---

## 9. Files Created

| File | Purpose |
|------|---------|
| `tools/governance/__init__.py` | Package init |
| `tools/governance/extract_features.py` | Feature extraction from fitz-gov cases (LLM-based) |
| `tools/governance/train_classifier.py` | Multi-model training with hyperparameter search |
| `tools/governance/data/features.csv` | 914 rows x 52 columns (constraint features) |
| `tools/governance/data/model_v1.joblib` | Best model artifact (RF tuned) + encoders + feature names |

---

## 10. Open Questions

1. ~~**Training approach**: Start with all ~40 features, or iteratively add feature tiers?~~ RESOLVED — used all Tier 1-2 features + 11 new context features = 58 total
2. ~~**Cross-validation strategy**: Stratified k-fold by subcategory? Or random?~~ RESOLVED — Stratified 5-fold by expected_mode
3. **Which model to ship?** RF (best accuracy) vs Ensemble (best dispute recall) vs GBT (balanced)
4. **Integration**: How to integrate classifier into the full pipeline? Replace AnswerGovernor.decide() or add as parallel path?
5. **Real pipeline evaluation**: Need to test with full retrieval pipeline data (vector scores, detection summaries) to get realistic accuracy numbers
6. **Dispute recall improvement**: Can we tune CA constraint sensitivity? Add more disputed training data? Two-stage prediction (RF + dispute refinement)?
7. **Fallback strategy**: Hard cutoff (classifier only) or soft (classifier + priority rules for low-confidence predictions)?

---

## 11. Next Steps

### Short-term (improve current results)
1. **Tune CA sensitivity** — CA only fired 221/914 cases. Many disputed cases lack the signal. Lower thresholds = more dispute features for classifier.
2. **More disputed training data** — 145 disputed cases vs 420 qualified. Add 50-100 more hard dispute cases.
3. **Two-stage model** — RF for general prediction, then dispute-specific check on cases predicted as "qualified" (catches the 16/29 D->Q confusions).

### Medium-term (integration)
4. **Integration prototype** — `fitz_ai/core/guardrails/classifier.py` wrapper that loads model_v1.joblib, runs at inference time.
5. **Full pipeline eval** — Run classifier on real retrieval results (not synthetic) to measure actual improvement over governor's ~70%.
6. **A/B comparison** — Run both governor and classifier on same queries, compare decisions.

### Longer-term
7. **Production deployment** — Ship model in package, load at engine startup, replace governor.
8. **Online learning** — Track predictions, collect corrections, retrain periodically.
9. **Confidence calibration** — Use classifier probabilities for soft decisions (high confidence = direct, low = fall back to priority rules).

---

## Changelog

| Date | Change |
|------|--------|
| 2025-02-08 | Initial document — findings from fitz-gov analysis + pipeline feature inventory |
| 2025-02-08 | Due diligence complete -- test case gap analysis + feature extraction audit |
| 2025-02-08 | Staging merge verified (was already merged). Subcategories consolidated: 156 -> 54 canonical types |
| 2025-02-08 | Generated 123 new cases (dispute boundary, edge cases, code/adversarial). Blind validated at 94% agreement. 4 relabeled. Merged into tier1_core. Total: 848 cases |
| 2026-02-08 | Feature extraction implementation complete (Tier 1-3). ~40 features flowing through pipeline. Verified with integration tests (22 passed). |
| 2026-02-08 | Feature extraction pipeline built (`extract_features.py`). 914 cases extracted with Cohere command-r7b, 1 worker, 0 errors, 52 columns. Governor baseline: 33.9%. |
| 2026-02-08 | Experiment 1 (baseline GBT): 57.4% accuracy vs 33.3% governor. Disputed recall: 28% (8/29). |
| 2026-02-08 | Experiment 2 (v2 — context features + class weighting + multi-model + hyperparam search): RF 71.0%, Ensemble 69% dispute recall. 11 new context features dominate importance. |
