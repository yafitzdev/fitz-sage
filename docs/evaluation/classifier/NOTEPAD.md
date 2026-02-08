# Governance Classifier — Living Notepad

**Goal**: Replace hand-coded `AnswerGovernor.decide()` priority rules with a trained tabular classifier.
**Status**: Feature extraction complete, ready for classifier training

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

## 7. Open Questions

1. ~~**Staging merge**: Is the staging data ready to merge?~~ RESOLVED — all data merged (848 tier1 cases)
2. ~~**Feature extraction**: What features are available and how to surface them?~~ RESOLVED — 40 features implemented across Tiers 1-3
3. **Training approach**: Start with all ~40 features, or iteratively add feature tiers?
4. **Cross-validation strategy**: Stratified k-fold by subcategory? Or random?
5. **Fallback strategy**: Hard cutoff (classifier only) or soft (classifier + priority rules for low-confidence predictions)?
6. ~~**When to generate more data**: Before first training run?~~ RESOLVED — generated 123 new cases, all validated and merged

---

## 8. Next Steps: Classifier Training

### Ready to train
- 848 labeled cases (tier1) + 44 tier0 sanity cases
- ~40 features extractable from the pipeline
- Feature extraction code complete and tested

### Training pipeline needed
1. **Feature matrix generation**: Run each test case through `extract_features()` to build X matrix
2. **Label vector**: Map `expected_mode` to 4-class target (abstain=0, disputed=1, qualified=2, confident=3)
3. **Train/test split**: 80/20 stratified by `expected_mode`
4. **Model selection**: XGBoost or scikit-learn GBT, 5-fold cross-validation
5. **Evaluation**: Per-class precision/recall, confusion matrix, feature importance
6. **Integration**: Replace `AnswerGovernor.decide()` with classifier inference

---

## Changelog

| Date | Change |
|------|--------|
| 2025-02-08 | Initial document — findings from fitz-gov analysis + pipeline feature inventory |
| 2025-02-08 | Due diligence complete -- test case gap analysis + feature extraction audit |
| 2025-02-08 | Staging merge verified (was already merged). Subcategories consolidated: 156 -> 54 canonical types |
| 2025-02-08 | Generated 123 new cases (dispute boundary, edge cases, code/adversarial). Blind validated at 94% agreement. 4 relabeled. Merged into tier1_core. Total: 848 cases |
| 2026-02-08 | Feature extraction implementation complete (Tier 1-3). ~40 features flowing through pipeline. Verified with integration tests (22 passed). |
