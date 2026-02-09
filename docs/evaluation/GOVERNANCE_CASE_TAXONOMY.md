# Governance Case Taxonomy

Complete taxonomy of governance case types for the fitz-gov benchmark.

**Status**: Complete. All cases merged into `fitz-gov/data/tier1_core/`. Current version: **v3.0.0**.

**Repository**: `C:\Users\yanfi\PycharmProjects\fitz-gov`

**Data structure** (after cleanup):
```
fitz-gov/data/
├── tier0_sanity/    60 sanity cases (baseline, models should score 95%+)
├── tier1_core/      1113 cases (1047 governance + 66 grounding/relevance)
├── corpus/          378 test documents
└── queries/         Query-to-document mappings for Mode B evaluation
```

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| Total cases (tier0 + tier1) | 1173 |
| Governance cases (abstain/dispute/qualify/confident) | 1091 (1047 tier1 + 44 tier0) |
| Other cases (grounding/relevance) | 82 |
| Unique subcategories (consolidated) | 54 |
| Total cases removed (duplicates) | 8 |
| Total cases relabeled (blind validation) | 20 |

### Per-Mode Distribution (tier1_core, governance only)

| Mode | Cases | % |
|------|-------|---|
| Qualification | 360 | 34.4% |
| Confidence | 254 | 24.3% |
| Abstention | 237 | 22.6% |
| Dispute | 196 | 18.7% |
| **Total** | **1047** | 100% |

Max:min class ratio is 2.2:1. Qualification remains largest but class balance improved significantly from v2.0 (was 2.9:1).

---

## Data Expansion History

### v1.0: Original 200 cases (from 21 experiments)

Hand-crafted cases derived from failure analysis across 21 evaluation experiments. These form the seed data for the benchmark.

| Category | Cases | Difficulty |
|----------|-------|-----------|
| Abstention | 40 | easy/medium |
| Dispute | 40 | easy/medium |
| Qualification | 40 | easy/medium |
| Confidence | 30 | easy/medium |
| Grounding | 25 | easy/medium |
| Relevance | 25 | easy/medium |
| **Total** | **200** | |

These 200 cases were split into tier0 (60 sanity, re-IDed as `t0_*`) and tier1 (141 baseline, re-IDed as `t1_*`).

### v2.0: +525 generated cases (LLM-assisted boundary sampling)

Generated using Claude to produce hard boundary cases across all subcategory types. Cases were organized into 7 generation batches targeting specific boundaries and mode pairs.

| Generation batch | Cases | Mode distribution | Subcategories | Target |
|-----------------|-------|-------------------|---------------|--------|
| Pure abstain + dispute | 90 | 50 abstain, 40 disputed | 18 | Core pure cases for both modes |
| Pure qualify + confident | 95 | 60 qualified, 35 confident | 19 | Core pure cases for both modes |
| D-Q boundary part 1 | 70 | 50 qualified, 20 disputed | 7 | Primary bottleneck: dispute vs qualify |
| D-Q boundary part 2 | 70 | 50 qualified, 10 disputed, 10 confident | 7 | D-Q boundary + confident edge |
| Abstain boundary | 65 | 40 abstain, 25 qualified | 13 | Abstain vs qualify/confident |
| Confident boundary | 45 | 35 confident, 10 qualified | 9 | Confident vs qualify |
| Three-way ambiguity | 90 | 66 qualified, 12 abstain, 12 confident | 13 | Multi-signal competition |
| **Total** | **525** | **102 abs, 70 disp, 261 qual, 92 conf** | | |

**Validation**: Independent blind label validation achieved 95.4% agreement. 5 duplicates removed, 7 cases relabeled (see Validation Results below). 513 cases merged into tier1_core (12 excluded as duplicates or cross-mode conflicts).

### v3.0: +123 generated cases (targeted gaps)

Targeted specific coverage gaps identified after v2.0 merge: dispute class underrepresentation, edge cases, and code/adversarial scenarios.

| Generation batch | Cases | Mode distribution | Target |
|-----------------|-------|-------------------|--------|
| Dispute boundary | 48 | 25 disputed, 13 qualified, 10 abstain | D-Q and A-D boundary expansion |
| Edge cases | 40 | 15 abstain, 10 mixed, 15 mixed | Empty context, short queries, long contexts |
| Code/adversarial | 35 | 20 mixed, 5 mixed, 10 mixed | Code/structured, multi-entity, adversarial |
| **Total** | **123** | | |

**Validation**: 94% blind label agreement. 4 cases relabeled (all disputed/abstain -> qualified). All 123 merged into tier1_core.

### v4.0: +199 generated cases (classifier failure-mode targeting)

Generated per `CLASSIFIER_V1_TEST_PLAN.md` to address specific classifier failure modes identified in Experiments 1-5. Focused on confident patterns (opposing_with_consensus, contradiction_resolved, different_framing), subtle disputes, and abstain edge cases.

| Generation batch | Cases | Mode distribution | Target |
|-----------------|-------|-------------------|--------|
| Confident patterns | 95 | 95 confident | Confident recall 48% → 70%+ |
| Subtle disputes | 60 | 60 disputed | Implicit contradiction, binary conflict, temporal conflict |
| Abstain edge cases | 45 | 45 abstain | Near-miss topics, entity confusion, temporal staleness |
| **Total** | **200** | **45 abs, 60 disp, 0 qual, 95 conf** | |

**Validation**: 93.5% blind label agreement (187/200). 9 cases fixed: 5 temporal supersession reclassified to confident, 3 metric-mismatch to qualified, 1 duplicate removed. 199 cases merged into tier1_core.

---

## Pure Cases

Cases where classification is unambiguous. These test basic signal recognition.

### Abstain — context doesn't answer the query (156 cases, 36 subcategories)

| Case type | Subcategory | Cases | Signal |
|-----------|-------------|-------|--------|
| Wrong entity entirely | `wrong_entity`, `wrong_entity_pure` | 14 | Different subject |
| Wrong domain | `wrong_domain` | 5 | Homonym mismatch |
| Wrong version | `wrong_version` | 5 | Version mismatch |
| Wrong jurisdiction | `wrong_jurisdiction` | 5 | Region mismatch |
| Wrong time period | `wrong_time_period` | 8 | Temporal staleness |
| Wrong specificity | `wrong_specificity` | 5 | Granularity mismatch |
| Wrong product | `wrong_product` | 5 | Adjacent product |
| Missing data entirely | `missing_data` | 5 | Information absence |
| Topic adjacent, no answer | `topic_adjacent` | 5 | Same system, wrong section |
| Format impossible | `format_impossible` | 5 | Can't satisfy format need |
| Decoy keywords | `decoy_keywords` | 11 | Shares vocabulary, different topic |
| Domain bleed | `domain_bleed` | 7 | Closely related field |
| Wrong aspect | `wrong_aspect` | 6 | Right entity, wrong dimension |
| Code abstention | `code_abstention` | 3 | Code-specific irrelevance |
| Cross-domain insufficient | `cross_domain_insufficient` | 3 | Concept transfer too weak for answer |

Also includes: `adjacent_product` (5), `high_similarity_wrong_entity` (5), `irrelevant_internal_tension` (5), `off_topic_contradicting` (6), `off_topic_contradiction` (5), `partial_schema_match` (5), `version_near_miss` (5), `wrong_jurisdiction_conflicts` (6), and 13 more low-count subcategories from pre-expansion cases.

### Dispute — sources make mutually exclusive factual claims (109 cases, 26 subcategories)

| Case type | Subcategory | Cases | Signal |
|-----------|-------------|-------|--------|
| Same metric, different values | `same_metric_different_values`, `same_claim_different_values` | 12 | Numerical conflict |
| Opposing conclusions | `opposing_conclusions`, `opposing_conclusions_genuine` | 15 | Direct opposition |
| Contradictory dates | `contradictory_dates` | 5 | Timeline conflict |
| Contradictory attribution | `contradictory_attribution` | 5 | Credit conflict |
| Contradictory status | `contradictory_status` | 4 | State conflict |
| Opposing recommendations | `opposing_recommendations` | 5 | Prescriptive conflict |
| Statistical direction conflict | `statistical_direction_conflict` | 5 | Trend conflict |
| Binary fact conflict | `binary_fact_conflict` | 5 | Boolean conflict |
| Implicit contradiction | `implicit_contradiction` | 11 | Implications conflict |
| Competing theories | `competing_theories` | 6 | Scientific disagreement |
| Conditional conflict | `conditional_conflict` | 6 | Context-dependent contradiction |

Also includes: `numerical_disagreement` (6), `temporal_conflict` (3), `source_conflict` (4), `methodological_conflict` (3), `scope_conflict` (2), and 10 more low-count subcategories.

### Qualify — answer requires caveats or hedging (318 cases, 56 subcategories)

| Case type | Subcategory | Cases | Signal |
|-----------|-------------|-------|--------|
| Same topic, different aspects | `same_topic_different_aspects`, `different_aspects` | 15 | Multi-faceted |
| Mixed evidence | `mixed_evidence` | 5 | Inconclusive |
| Conditional applicability | `conditional_applicability` | 5 | Scoped validity |
| Hedged/uncertain claims | `hedged_claims`, `hedged_source` | 8 | Epistemic weakness |
| Temporal ambiguity | `temporal_ambiguity` | 7 | Freshness uncertain |
| Entity ambiguity | `entity_ambiguity` | 8 | Referent unclear |
| Scope ambiguity | `scope_ambiguity`, `scope_ambiguity_pure` | 8 | Underspecified reference |
| Deprecated but documented | `deprecated_documented` | 5 | Valid with caveat |
| Partial correlation | `partial_correlation` | 5 | Causal uncertainty |
| Small sample / weak methodology | `small_sample`, `small_sample_weak` | 9 | Evidence quality concern |
| Source quality variance | `source_quality_variance`, `source_quality` | 7 | Authority mismatch |
| Multiple valid interpretations | `multiple_interpretations` | 5 | Metric ambiguity |
| Methodology difference | `methodology_difference`, `methodology_difference_relabeled` | 14 | Measurement approach differs |
| Hedged vs assertive | `hedged_vs_assertive` | 10 | Asymmetric evidence strength |
| Numerical near-miss | `numerical_near_miss` | 10 | Rounding, not conflict |
| Same claim, different conditions | `same_claim_different_conditions` | 10 | Both true in different contexts |
| Same claim, different time periods | `same_claim_different_timeperiods` | 10 | Both true at different times |
| Evolving facts | `evolving_facts` | 9 | Superseded, not contradicted |
| Source quality asymmetry | `source_quality_asymmetry` | 10 | Study vs anecdote disagree |
| Pros vs cons | `pros_cons_same_thing` | 10 | Balanced assessment |
| Risk vs benefit | `risk_vs_benefit` | 10 | Both true simultaneously |
| Correlation / causation | `correlation_causation` | 8 | Causal leap |

Also includes: `adjacent_entity_overlap` (5), `partial_answer` (5), `related_missing_specific` (5), `right_topic_wrong_infotype` (5), `tangential_useful` (5), `implicit_assumptions` (5), `old_likely_valid` (5), `cross_domain_transfer` (4), `prediction_insufficient_data` (6), and 18 more subcategories from pre-expansion and three-way cases.

### Confident — clear, consistent evidence (142 cases, 38 subcategories)

| Case type | Subcategory | Cases | Signal |
|-----------|-------------|-------|--------|
| Direct factual answer | `direct_factual`, `direct_factual_pure` | 5 | Explicit answer |
| Multiple sources converge | `multi_source_convergence`, `multi_source_convergence_pure`, `multi_source_agreement` | 11 | Corroborated |
| Clear procedural answer | `clear_procedural`, `procedural_complete` | 7 | Complete procedure |
| Unambiguous extraction | `unambiguous_extraction`, `table_extraction` | 9 | Structured data match |
| Well-documented technical | `well_documented_technical` | 5 | Authoritative source |
| Clear causal explanation | `clear_causal_explanation`, `explicit_causal` | 10 | Complete explanation |
| Quantitative answer available | `quantitative_available`, `quantitative_clear` | 6 | Exact match |
| Different framing, same fact | `different_framing_same_fact` | 10 | Apparent contradiction resolves |
| Opposing with consensus | `opposing_with_consensus` | 6 | Overwhelming agreement |
| Numerical diff, methodology explained | `numerical_diff_methodology_explained` | 5 | Difference is explained |
| Clear answer, minor edge case | `clear_answer_minor_edge` | 5 | Exception doesn't matter |
| Single authoritative source | `single_authoritative` | 5 | Definitive without corroboration |
| Contradiction with clear winner | `contradiction_clear_winner` | 5 | One source obviously wrong |

Also includes: `apparent_contradiction_granularity` (5), `minor_disagreement_clear_answer` (5), `near_complete_evidence` (5), `slight_variation_same_answer` (5), `code_documentation` (3), `api_confidence` (3), and 15 more subcategories.

---

## Boundary Cases

Cases where classification is hard. These are the highest-value test cases and were the primary focus of the expansion.

### Abstain <-> Confident (11 subcategories)

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Decoy keywords | abstain | `decoy_keywords` | 11 | Shares vocabulary, different topic |
| Domain bleed | abstain | `domain_bleed` | 7 | Closely related field |
| Adjacent product | abstain | `adjacent_product` | 5 | Same product line, wrong model |
| Version near-miss | abstain | `version_near_miss` | 5 | Same software, wrong version |
| Partial schema match | abstain | `partial_schema_match` | 5 | Table has some columns, missing key one |
| High embedding similarity, wrong entity | abstain | `high_similarity_wrong_entity` | 5 | Embeddings can't distinguish |

### Abstain <-> Qualify (5 subcategories)

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Related topic, missing specific info | qualify | `related_missing_specific` | 5 | Context IS about the topic |
| Partial answer | qualify | `partial_answer` | 5 | Has 1 of 3 things asked about |
| Adjacent entity, some overlap | qualify | `adjacent_entity_overlap` | 5 | Some specs shared |
| Right topic, wrong info type | qualify | `right_topic_wrong_infotype` | 5 | Gets features, asked for pricing |
| Tangential but useful context | qualify | `tangential_useful` | 5 | Relevant background, no direct answer |

### Abstain <-> Dispute (2 subcategories)

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Irrelevant content with internal tension | abstain | `irrelevant_internal_tension` | 5 | Chunks contradict but don't answer |
| Off-topic contradiction | abstain | `off_topic_contradiction` | 5 | Real contradiction, wrong subject |

### Dispute <-> Qualify (14 subcategories — primary bottleneck)

The densest boundary in the taxonomy. 140+ cases across both dispute and qualify sides.

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Same claim, different values | dispute | `same_claim_different_values` | 7 | Factual conflict on same measurement |
| Opposing conclusions (genuine) | dispute | `opposing_conclusions_genuine` | 10 | Genuine disagreement |
| Implicit contradiction | dispute | `implicit_contradiction` | 11 | Implications conflict |
| Same topic, different aspects | qualify | `same_topic_different_aspects` | 10 | Complementary info misread as conflict |
| Same claim, different time periods | qualify | `same_claim_different_timeperiods` | 10 | Both true at different times |
| Same claim, different conditions | qualify | `same_claim_different_conditions` | 10 | Both true in different contexts |
| Methodology difference | qualify | `methodology_difference` | 14 | Measurement approach differs |
| Hedged vs assertive | qualify | `hedged_vs_assertive` | 10 | Asymmetric evidence strength |
| Numerical near-miss | qualify | `numerical_near_miss` | 10 | $5.0M vs $5.2M — methodology |
| Pros vs cons | qualify | `pros_cons_same_thing` | 10 | Balanced assessment |
| Risk vs benefit | qualify | `risk_vs_benefit` | 10 | Both true simultaneously |
| Evolving facts | qualify | `evolving_facts` | 9 | Superseded, not contradicted |
| Source quality asymmetry | qualify | `source_quality_asymmetry` | 10 | Study vs anecdote disagree |
| Different framing, same fact | confident | `different_framing_same_fact` | 10 | Apparent contradiction resolves |

### Qualify <-> Confident (6 subcategories)

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Clear answer with minor edge case | confident | `clear_answer_minor_edge` | 5 | Exception exists but main answer clear |
| Single authoritative source | confident | `single_authoritative` | 5 | Definitive without corroboration |
| Old but likely still valid | qualify | `old_likely_valid` | 5 | Probably correct but undated |
| Answer with implicit assumptions | qualify | `implicit_assumptions` | 5 | Correct IF conditions hold |
| Near-complete evidence | confident | `near_complete_evidence` | 5 | 95% present, minor detail missing |
| Multiple sources, slight variation | confident | `slight_variation_same_answer` | 5 | Phrasing differs, substance matches |

### Dispute <-> Confident (3 subcategories)

| Case type | Correct mode | Subcategory | Cases | Why it's confusing |
|-----------|--------------|-------------|-------|--------------------|
| Apparent contradiction, different granularity | confident | `apparent_contradiction_granularity` | 5 | "Revenue grew" vs "Q3 dipped" — both true |
| Contradiction with clear winner | confident | `contradiction_clear_winner` | 5 | One source is clearly wrong |
| Minor disagreement in clear answer | confident | `minor_disagreement_clear_answer` | 5 | Noise, not signal |

---

## Three-Way Ambiguity Cases

Cases where multiple signals compete. These are the hardest cases and where hand-tuned priority rules break down. Generated as 90 cases across 13 subcategories.

### Dispute <-> Qualify <-> Confident

| Case type | Correct mode | Subcategory | Cases | Competing signals |
|-----------|--------------|-------------|-------|-------------------|
| Evolving facts with source quality | qualify | `evolving_facts_source_quality` | 8 | Old study vs new blog — dispute/qualify/confident |
| Numerical diff, methodology explained | confident | `numerical_diff_methodology_explained` | 5 | Different values but explanation resolves it |
| Opposing with consensus | confident | `opposing_with_consensus` | 6 | 9 studies say X, 1 says Y |
| Hedged contradiction, corroborated | qualify | `hedged_contradiction_corroborated` | 8 | "X may cause Y" vs "X does not" + corroboration |

### Abstain <-> Qualify <-> Confident

| Case type | Correct mode | Subcategory | Cases | Competing signals |
|-----------|--------------|-------------|-------|-------------------|
| Adjacent version with overlap | qualify | `adjacent_version_overlap` | 7 | Wrong version but 80% same API |
| Stale authoritative source | qualify | `stale_authoritative` | 7 | Outdated but was authoritative |
| Partial answer from definitive source | qualify | `partial_answer_definitive` | 7 | Partial but what's there is authoritative |
| Cross-domain transfer | qualify | `cross_domain_transfer` | 4 | Concepts transfer across domains |

### Abstain <-> Dispute <-> Qualify

| Case type | Correct mode | Subcategory | Cases | Competing signals |
|-----------|--------------|-------------|-------|-------------------|
| Off-topic contradicting sources | abstain | `off_topic_contradicting` | 6 | Real contradiction about wrong entity |
| Wrong jurisdiction with conflicts | abstain | `wrong_jurisdiction_conflicts` | 6 | Wrong jurisdiction but real conflict |
| Version mismatch with breaking changes | qualify | `version_mismatch_breaking` | 7 | Wrong version but migration guide is relevant |

### Full Four-Way Ambiguity

| Case type | Correct mode | Subcategory | Cases | Competing signals |
|-----------|--------------|-------------|-------|-------------------|
| Adjacent entity, contradictory, hedged | qualify | `adjacent_entity_contradictory_hedged` | 8 | Wrong entity + contradiction + hedging + clinical data |
| Stale contradictory, partial coverage | qualify | `stale_contradictory_partial` | 6 | Outdated + contradiction + partial + trend |

---

## Validation Results

### Duplicate Removal (5 cases)

| Removed ID | Reason |
|------------|--------|
| `t1_qualify_hard_332` | Exact duplicate of `t1_qualify_hard_215` (EV market share Norway) |
| `t1_confident_hard_104` | Cross-mode conflict with `t1_dispute_hard_117` (WWW invention) |
| `t1_confident_hard_132` | Cross-mode conflict with existing `t1_abstain_medium_004` (Tokyo population) |
| `t1_confident_hard_605` | Same-mode duplicate of existing `t1_confident_hard_030` (speed of light) |
| `t1_qualify_hard_660` | Cross-mode conflict with existing `t1_abstain_medium_013` (Bitcoin price) |

### Independent Blind Label Validation (7 relabeled)

A separate Claude instance labeled all 525 cases without seeing the original labels. 95.4% agreement. 7 firm disagreements, all accepted and relabeled:

| Original ID | Original | Relabeled | New ID | Pattern |
|-------------|----------|-----------|--------|---------|
| `t1_dispute_hard_200` | disputed | qualified | `t1_qualify_hard_700` | Scope difference (as-reported vs pro forma) |
| `t1_dispute_hard_205` | disputed | qualified | `t1_qualify_hard_701` | Scope difference (direct vs total cost) |
| `t1_dispute_hard_206` | disputed | qualified | `t1_qualify_hard_702` | Methodology difference (count vs mass) |
| `t1_dispute_hard_120` | disputed | qualified | `t1_qualify_hard_703` | Semantic ambiguity, not factual contradiction |
| `t1_qualify_hard_630` | qualified | abstain | `t1_abstain_hard_704` | Wrong language (JS/Node/C# for Python query) |
| `t1_qualify_hard_634` | qualified | abstain | `t1_abstain_hard_705` | Wrong country (Australia/Argentina for Chile) |
| `t1_qualify_hard_635` | qualified | abstain | `t1_abstain_hard_706` | Wrong platform (GitHub/Jenkins/Azure for GitLab) |

### v3.0 Blind Label Validation (4 relabeled)

123 new cases across 3 files, validated at 94% agreement. 4 relabeled:

| Original ID | Original | Relabeled | Pattern |
|-------------|----------|-----------|---------|
| `t1_dispute_hard_407` | disputed | qualified | Methodology difference (EPA vs ACC plastic recycling definitions) |
| `t1_dispute_hard_419` | disputed | qualified | Different metrics (median hourly FT vs total annual all-workers pay gap) |
| `t1_abstain_hard_867` | abstain | qualified | Engineering practices ARE compliance controls under SOC 2/ISO 27001 |
| `t1_dispute_hard_506` | disputed | qualified | False premise with unanimous refutation, not a dispute between sources |

**v3.0 expansion categories:**
- Dispute boundary cases (48): D-Q boundary (methodology_difference vs genuine conflict), A-D boundary (off-topic contradiction)
- Edge cases (40): empty context (15), short queries (10), long contexts (15)
- Code/adversarial (35): code/structured data (20), multi-entity comparison (5), adversarial queries (10)

### Key Validation Findings

1. **Dispute vs qualify at methodology/scope boundary**: The hardest labeling decision. When sources report different numbers because they measure different things (pro forma vs as-reported, count vs mass), this is qualify (methodology difference), not dispute. The validator established the rule: if the gap is FULLY EXPLAINED by a stated methodology/scope difference, it should be qualified.

2. **Cross-domain transfer boundary**: Concept transfer from one language/platform/country to another does not constitute a partial answer when the query asks about a specific target. Python async != JavaScript async. Chile mining law != Argentina mining law. Three cases were reclassified from qualify to abstain.

3. **Strongest subcategories** (zero disagreements): `wrong_entity_pure`, `wrong_domain`, `version_near_miss`, `domain_bleed`, `same_topic_different_aspects`, `same_claim_different_timeperiods`, `same_claim_different_conditions`, `opposing_conclusions_genuine`.

---

## Coverage Summary (post v3.0 expansion)

| Region | Cases | Notes |
|--------|-------|-------|
| Pure: Abstain | 192 | +36 from edge/code cases |
| Pure: Dispute | 145 | +36 from boundary expansion |
| Pure: Qualify | 357 | +39 from boundary relabels + edge/code |
| Pure: Confident | 154 | +12 from edge/code cases |
| Boundary: Dispute <-> Qualify | ~175 | Primary bottleneck, well-covered |
| Boundary: Abstain <-> Dispute | ~20 | Expanded from 10 |
| Empty context | 15 | New in v3.0 |
| Short queries (<20 chars) | 10 | New in v3.0 |
| Long contexts (1500+ chars) | 15 | New in v3.0 |
| Code/structured data | 20 | New in v3.0 |
| Adversarial/trick queries | 10 | New in v3.0 |
| Multi-entity comparison | 5 | New in v3.0 |

**Subcategories consolidated**: 156 raw slugs -> 54 canonical types via `scripts/consolidate_subcategories.py`. Only 3 subcategories have <5 cases.
