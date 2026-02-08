# fitz-gov 2.0 Benchmark Results

**Version**: fitz-gov 2.0.0 (331 test cases)
**Date**: February 6, 2026 (initial), February 8, 2026 (current)
**Model**: Ollama qwen2.5:3b (local)
**Branch**: `refactor/staged-constraint-pipeline`

---

## Executive Summary

fitz-gov 2.0 expands the benchmark from 220 to 331 test cases, adding challenging categories for code context, ambiguous queries, structured data, and edge cases.

### Current Results (Governance, 249 cases)

After 18 optimization experiments (see RESEARCH_NOTEPAD.md):

| Metric | Initial (Feb 6) | Current Range | Δ |
|--------|-----------------|---------------|-----|
| **Governance Overall** | **63.14%** | **71-72%** | **+8-9%** |
| **Abstention** | 57.14% | 57.1% | ~0% |
| **Dispute Detection** | 89.09% | 87-89% | -1 to 0% |
| **Qualification** | 47.06% | 56-79% | **+9 to +32%** |
| **Confidence** | 79.37% | 86-89% | **+7 to +10%** |

*Note*: Initial 63.14% was on all 331 cases. Current ~71% is on 249 governance cases (excludes grounding/relevance which are tested separately). Qualification shows high LLM variance (56-79%) due to conflict_aware nondeterminism — the 3b model gives inconsistent pairwise contradiction answers, swinging 16+ qualification cases between runs. Overall score is stable at 71-72% because qualification swings partially offset each other.

### Relevance Mode Classification (40 cases)

| Metric | Peak (Exp 004) | After IE Fix (Exp 019) | Δ from Peak |
|--------|----------------|------------------------|-------------|
| **Relevance** | **67.5%** (27/40) | **35.0%** (14/40) | **-32.5%** |

Relevance was at 22.5% (9/40) due to IE `missing_primary` over-firing on ALL-CAPS query-intent words. Exp 019 (entity extraction fix) recovered to 35.0% (14/40) by:
1. Skipping ALL-CAPS words as emphasis markers, not proper nouns
2. Expanding generic_words to exclude query-aspect words (pricing, deadline, warranty, etc.)
3. Filtering multi-word LLM candidates where all words are generic

Remaining 26 failures are all `qualified->confident` (21 cases) and `qualified->abstain` (5 cases). The confident failures need a new sufficiency constraint — context IS topically related but doesn't contain the specific information asked about. See RESEARCH_NOTEPAD.md Exp 019 for full analysis.

### Grounding (42 cases) — Answer Quality Test

| Metric | Score | Correct/Total |
|--------|-------|---------------|
| **Grounding** | **90.5%** | **38/42** |

*Note*: Grounding tests **answer text quality** (forbidden claim detection in generated text), not governance mode classification. Evaluated via `run_targeted_benchmark.py --grounding` which generates LLM answers and checks for forbidden claims. 4 failures: 2 legit hallucinations (table_inference: model computed values from table data), 2 potential fitz-gov test case issues (code_grounding: forbidden patterns match words in "I cannot find" responses).

### Version Comparison

| Metric | v1.0 (200 cases) | v2.0 Initial | v2.0 Current |
|--------|------------------|--------------|--------------|
| Governance | 72.0% | 63.14% | ~71% |
| Abstention | 72.5% | 57.14% | 57.1% |
| Dispute | 90.0% | 89.09% | ~89% |
| Qualification | 72.5% | 47.06% | 56-79%* |
| Confidence | 86.67% | 79.37% | ~86% |

*Qualification has high LLM variance (see note above).

---

## Test Set Expansion

### What's New in v2.0

| Category | v1.0 | v2.0 | New Subcategories |
|----------|------|------|-------------------|
| Abstention | 40 | 63 (+23) | `code_abstention`, `table_absence`, `temporal_staleness` |
| Dispute | 40 | 55 (+15) | `time_dependent_contradiction`, `unit_scale_mismatch` |
| Qualification | 40 | 68 (+28) | `deprecation_qualification`, `entity_ambiguity`, `metric_ambiguity` |
| Confidence | 30 | 63 (+33) | `api_confidence`, `table_extraction`, `json_navigation` |
| Grounding | 25 | 42 (+17) | `code_grounding`, `table_inference` |
| Relevance | 25 | 40 (+15) | `format_mismatch`, `granularity_mismatch` |
| **Total** | **200** | **331** | **+131 cases** |

### New Test Categories

1. **Code Context** - API documentation, function signatures, deprecation warnings
2. **Ambiguous Queries** - Multiple valid interpretations requiring qualification
3. **Structured Data** - Tables, JSON navigation, missing columns
4. **Edge Cases** - Temporal staleness, jurisdictional mismatches, domain bleed

---

## Detailed Results

### Confusion Matrix Analysis

```
              abstain  disputed  qualified  confident
   abstain        36         8          0         19
  disputed         2        49          2          2
 qualified         3        26         32          7
 confident         4         6          3         50
```

### Primary Failure Modes (Current)

| Failure Pattern | Initial | Current | Impact |
|-----------------|---------|---------|--------|
| **Qualified→Disputed** | 26 | 19 | Over-detecting contradictions (conflict_aware, 3b limit) |
| **Abstain→Confident** | 19 | 16 | Decoy data — topically similar but wrong entity |
| **Abstain→Disputed** | 8 | 8 | Finding contradictions in irrelevant content |
| **Qualified→Confident** | 7 | 7 | Missing uncertainty that requires hedging |
| **Confident→Disputed** | 6 | 5 | False contradiction detection |

### Performance by Category (Current)

#### Governance Modes

| Category | Initial | Current | Main Failure |
|----------|---------|---------|--------------|
| Abstention | 57.14% | 57.1% (36/63) | → Confident (decoy data) |
| Dispute | 89.09% | ~89% (49/55) | Stable |
| Qualification | 47.06% | 56-79% (38-54/68) | → Disputed (CA false fires, high variance) |
| Confidence | 79.37% | ~86% (54/63) | → Disputed (CA false fires) |

#### Relevance (Mode Classification)

| Metric | Peak (Exp 004) | After IE Fix (Exp 019) | Main Failure |
|--------|----------------|------------------------|--------------|
| Relevance | 67.5% (27/40) | 35.0% (14/40) | Nothing fires on topically-related-but-insufficient context |

#### Answer Quality (Requires Full Generation)

| Category | Accuracy | Correct/Total | Analysis |
|----------|----------|---------------|----------|
| Grounding | 90.5% | 38/42 | Strong - 2 legit hallucinations, 2 potential test case issues |

---

## Optimization Journey (63.14% → ~71%)

See RESEARCH_NOTEPAD.md for full experiment details.

| Exp | Change | Result |
|-----|--------|--------|
| 001-006 | IE staged pipeline, thresholds, primary entity extraction | 63.14% → 66.3% |
| 007-009 | CA model scaling analysis, SIT rate detection | 66.3% → 68.3% |
| 010-011 | Dispute subordination, qualified consensus rule | 68.3% → 70.3% |
| 012-017 | CA deep-dive (5 approaches), SIT/aspect LLM verifiers | All blocked by 3b limit |
| 018 | Causal attribution regex tightening | 70.3% → 71.5% |
| **019** | **IE entity extraction fix (ALL-CAPS, generic words)** | **Relevance: 22.5% → 35.0%** |

### Remaining Bottlenecks

1. **conflict_aware false fires (34 cases)** — Needs 7b+ model. Blocked.
2. **Decoy data (16 cases)** — Needs entity-level LLM verification. Blocked.
3. **Relevance sufficiency gap (21 cases)** — Needs new sufficiency constraint.
4. **Scattered (18 cases)** — Partially fixable with regex. Est. +2-3 cases.

### Theoretical Ceiling

- **With qwen2.5:3b**: ~73-75% governance (current 71.5% + scattered fixes)
- **With 7b+ for conflict_aware**: 75-80%+ (removes discrimination bottleneck)
- **Relevance**: ~50-55% (35% + sufficiency constraint for 21 confident failures)

### Notes

- Grounding is NOT a governance mode test. Evaluated via `--grounding` flag (text generation + forbidden claim check).
- Qualification has high LLM variance (56-79%) due to CA nondeterminism with 3b model.

---

## Reproduction

```bash
# Install fitz-gov 2.0
pip install fitz-gov==2.0.0

# Run governance benchmark (249 cases, ~2 min)
python run_targeted_benchmark.py

# Full benchmark with relevance (289 cases, ~2 min)
python run_targeted_benchmark.py --full

# Grounding text quality test (42 cases, ~1 min)
python run_targeted_benchmark.py --grounding
```

### Configuration Used

```python
# Governance constraints (run_targeted_benchmark.py)
constraints = [
    InsufficientEvidenceConstraint(chat=fast_chat, embedder=embedder),
    SpecificInfoTypeConstraint(),
    CausalAttributionConstraint(),
    ConflictAwareConstraint(chat=fast_chat, use_fusion=True, adaptive=True, embedder=embedder),
]
```

### Timing

- **Governance run** (249 cases): ~120s (~2 min)
- **Full run** (289 cases): ~120s (~2 min)
- **Grounding run** (42 cases): ~60s (~1 min)

---

## Conclusion

fitz-gov 2.0 expanded the benchmark from 200 to 331 cases, initially dropping accuracy from 72% to 63.14%. Through 19 experiments, governance accuracy recovered to **71.5%** (178/249 governance cases).

**Key takeaways**:
- Qualification accuracy saw the largest improvement: 47.06% → 56-79% (high variance due to CA nondeterminism)
- Confidence accuracy improved: 79.37% → ~84%
- Dispute detection remains strong (~89%)
- Grounding remains strong (90.5%) — tested via `--grounding` flag (text quality, not mode)
- Relevance **partially recovered**: 22.5% → 35.0% (Exp 019, IE entity extraction fix)
- Remaining relevance gap (21 cases) needs a sufficiency constraint
- Abstention remains the hardest category (~54%), blocked by decoy data problem
- The 3b model is the primary bottleneck — conflict_aware false fires (34 cases) and decoy data (16 cases) need larger model discrimination
- Estimated 3b ceiling: 73-75% governance. Model upgrade path: 75-80%+
- **Next priority**: Sufficiency constraint for relevance (21 cases where context is topically related but doesn't contain the specific info asked about)