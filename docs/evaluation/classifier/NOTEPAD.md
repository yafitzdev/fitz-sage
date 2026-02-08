# Governance Classifier — Living Notepad

**Goal**: Replace hand-coded `AnswerGovernor.decide()` priority rules with a trained tabular classifier.
**Status**: Research & planning phase

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

### Usable cases: 725 tier1 + 44 tier0 = 769 total

| Category | Tier 0 | Tier 1 | Total | `expected_mode` | Usable? |
|----------|--------|--------|-------|-----------------|---------|
| Abstention | 12 | 156 | 168 | abstain | Yes |
| Confidence | 10 | 142 | 152 | confident | Yes |
| Dispute | 12 | 109 | 121 | disputed | Yes |
| Qualification | 10 | 318 | 328 | qualified | Yes |
| Grounding | 8 | 34 | 42 | qualified* | No — tests answer quality, not mode |
| Relevance | 8 | 32 | 40 | qualified* | No — tests answer content, not mode |

*Grounding/relevance cases evaluate the generated answer text (regex + forbidden claims), not the governance mode decision. Including them would add noise.

**Staging data**: 525 cases were generated and merged (5 duplicates removed, 7 relabeled per validation report). The tier1 counts above reflect the post-merge state.

### Class distribution (725 tier1 cases)

```
qualified:  318 (43.9%)
abstain:    156 (21.5%)
confident:  142 (19.6%)
disputed:   109 (15.0%)
```

Imbalance is moderate (2.9x max/min). Workable without oversampling.

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

- 769 examples, ~25 features → textbook tabular classification
- No neural networks, no GPU, no deep learning
- scikit-learn or XGBoost — training is 3-5 lines of code
- Model size: KB-range, microsecond inference
- Handles mixed numeric/categorical features natively
- Built-in feature importance (explains decisions)
- Cross-validation works well at this scale

---

## 6. Due Diligence

### 6a. Test Case Gaps

#### Staging data: 525 validated cases waiting to merge

fitz-gov has `data/staging/` with 525 additional cases at 95.4% agreement (7 disagreements to relabel). Merging brings total from 769 → ~1,250 usable cases. This is the single highest-ROI action.

| Class | Current (tier1) | After staging merge | Impact |
|-------|-----------------|---------------------|--------|
| abstain | 156 | ~260 | Comfortable |
| confident | 142 | ~245 | Comfortable |
| disputed | 109 | ~179 | Crosses minimum threshold |
| qualified | 318 | ~579 | Large |

#### Statistical concern: disputed class is thin

With 80/20 split on current 109 disputed cases → ~21 test cases. Each misclassification moves class accuracy by 4.8%. Cannot reliably distinguish 80% from 90% accuracy with 21 samples (need ~30+). After staging merge (179 → ~35 test), this barely crosses the minimum.

**Decision needed**: Generate 30-50 more disputed cases focused on the Dispute↔Qualify boundary, or accept the statistical limitation.

#### Subcategory fragmentation

156 unique subcategories across 725 tier1 cases. Many are slug variants of the same concept:

| Problem | Count | Impact |
|---------|-------|--------|
| Subcategories with 1 case | 14 | Cannot generalize — memorized or misclassified |
| Subcategories with 2 cases | 25 | Nearly as bad |
| Subcategories with <5 cases | 64 | Thin coverage |

Worst offender: **confident** — 22 of 38 subcategories (58%) have <5 cases. Slug variants like `authoritative_source`/`official_statement`/`single_authoritative` fragment the data.

**Action**: Consolidate slug variants before training. Merge into ~50-60 canonical subcategories. No new data needed, just relabeling.

#### Boundary pair coverage vs taxonomy targets

| Boundary pair | Actual | Target | Gap |
|---------------|--------|--------|-----|
| Dispute ↔ Qualify | 144 | 210 | 66 |
| Abstain ↔ Confident | 38 | 60 | 22 |
| Qualify ↔ Confident | 35 | 60 | 25 |
| Abstain ↔ Qualify | 28 | 50 | 22 |
| Dispute ↔ Confident | 18 | 30 | 12 |
| **Abstain ↔ Dispute** | **10** | **20** | **10** |
| Three-way ambiguity | 74 | 110 | 36 |
| Four-way ambiguity | 14 | 20 | 6 |
| **Total boundary** | **361** | **560** | **199** |

Staging data closes most of these gaps. Abstain↔Dispute boundary is thinnest (10 cases, 2 subcategory types).

#### What's missing entirely

| Gap | Cases | Priority |
|-----|-------|----------|
| Empty/null context (retrieval returns nothing) | 0 | P1 — real systems hit this |
| Ultra-short queries ("Revenue?" "Status?") | 7 under 25 chars | P2 |
| Very long contexts (3000+ chars per chunk) | 0 | P2 |
| Code/structured data governance | 6 total | P2 — important for a RAG product |
| Adversarial/trick queries (false premises, leading) | ~2 | P3 |
| Multi-language | 0 | P3 — low priority unless needed |
| Multi-entity queries ("Compare X and Y" where only X has data) | 0 | P2 |

#### Difficulty distribution concern

93% of tier1 is "hard" difficulty. Almost no medium baseline. This makes the benchmark aspirational but not diagnostic — a model scoring 70% might actually be quite good, but there's no easy tier to confirm basic sanity beyond the 44 tier0 cases.

#### Priority action items

**P0 — Before training:**
1. Merge staging data (525 cases, already validated)
2. Consolidate slug fragmentation (~50-60 canonical subcategories)

**P1 — High ROI:**
3. Generate 40-50 more disputed boundary cases (Dispute↔Qualify focus)
4. Generate 20+ Abstain↔Dispute cases (thinnest boundary)
5. Add 15-20 empty-context cases (what happens when retrieval finds nothing?)

**P2 — Coverage:**
6. Expand all singleton subcategories to 5+ cases or merge them
7. Add ultra-short query cases, long-context cases, code governance cases

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

---

## 7. Open Questions

1. **Staging merge**: Is the staging data ready to merge? What are the 7 disagreements?
2. **Training approach**: Start with all 42 features, or iteratively add feature tiers?
3. **Cross-validation strategy**: Stratified k-fold by subcategory? Or random?
4. **Fallback strategy**: Hard cutoff (classifier only) or soft (classifier + priority rules for low-confidence predictions)?
5. **When to generate more data**: Before first training run, or after seeing what the classifier struggles with?

---

## 8. Implementation Plan

*TODO: After decisions on open questions*

---

## Changelog

| Date | Change |
|------|--------|
| 2025-02-08 | Initial document — findings from fitz-gov analysis + pipeline feature inventory |
| 2025-02-08 | Due diligence complete -- test case gap analysis + feature extraction audit |
| 2025-02-08 | Staging merge verified (was already merged). Subcategories consolidated: 156 -> 54 canonical types |
