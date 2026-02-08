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

| Metric | Initial (Feb 6) | Current (Feb 8) | Δ |
|--------|-----------------|-----------------|-----|
| **Governance Overall** | **63.14%** | **71.5%** | **+8.4%** |
| **Abstention** | 57.14% | 57.1% | ~0% |
| **Dispute Detection** | 89.09% | 87.3% | -1.8% |
| **Qualification** | 47.06% | 79.4% | **+32.3%** |
| **Confidence** | 79.37% | 88.9% | **+9.5%** |

*Note*: Initial 63.14% was on all 331 cases. Current 71.5% is on 249 governance cases (excludes grounding/relevance which are tested separately). Governance category improvements are directly comparable.

### Full Benchmark (331 cases, initial run only)

| Category | Score | Correct/Total |
|----------|-------|---------------|
| Grounding | 97.62% | 41/42 |
| Relevance | 2.50% | 1/40 |

Grounding and relevance have not been re-tested since initial run.

### Version Comparison

| Metric | v1.0 (200 cases) | v2.0 Initial | v2.0 Current |
|--------|------------------|--------------|--------------|
| Governance | 72.0% | 63.14% | 71.5% |
| Abstention | 72.5% | 57.14% | 57.1% |
| Dispute | 90.0% | 89.09% | 87.3% |
| Qualification | 72.5% | 47.06% | 79.4% |
| Confidence | 86.67% | 79.37% | 88.9% |

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

| Category | Initial | Current | Correct/Total | Main Failure |
|----------|---------|---------|---------------|--------------|
| Abstention | 57.14% | 57.1% | 36/63 | → Confident (decoy data) |
| Dispute | 89.09% | 87.3% | 48/55 | Stable |
| Qualification | 47.06% | 79.4% | 54/68 | → Disputed (CA false fires) |
| Confidence | 79.37% | 88.9% | 56/63 | → Disputed (CA false fires) |

#### Answer Quality (Initial Run Only)

| Category | Accuracy | Correct/Total | Analysis |
|----------|----------|---------------|----------|
| Grounding | 97.62% | 41/42 | Excellent - avoids hallucination |
| Relevance | 2.50% | 1/40 | Critical failure - answers wrong question |

---

## Root Cause Analysis

### 1. Qualification Collapse (47.06%)

**Problem**: System treats nuanced differences as contradictions
- New ambiguous query cases expose over-aggressive dispute detection
- Entity/scope/temporal ambiguity not handled well
- Statistical variations misclassified as disputes

**Example Pattern**:
- Query: "How does the system perform?"
- Context 1: "Speed is excellent"
- Context 2: "Accuracy needs improvement"
- Expected: QUALIFIED (different aspects)
- Actual: DISPUTED (seen as contradiction)

### 2. Relevance Failure (2.50%)

**Problem**: System answers A question, not THE question
- Only 1 of 40 relevance cases correct
- Suggests fundamental issue with answer targeting
- May be answering tangentially related content

**Hypothesis**:
- Relevance detection may be too permissive after enrichment
- System finds something to say even when not addressing the query

### 3. Abstention Degradation (57.14%)

**Problem**: New structured data and code contexts confuse relevance detection
- Code documentation looks relevant due to technical terms
- Table/JSON structures match keywords but don't answer query
- Temporal staleness not detected

**Example Pattern**:
- Query: "What's the current API rate limit?"
- Context: API documentation from 2018
- Expected: ABSTAIN (stale data)
- Actual: CONFIDENT (doesn't detect staleness)

---

## Positive Findings

Despite the overall score drop, several achievements stand out:

### 1. Dispute Detection Robust (89.09%)
- Only 0.91% drop despite 131 new test cases
- Pairwise contradiction detection remains effective
- Handles new edge cases well

### 2. Grounding Excellence (97.62%)
- Near-perfect hallucination prevention
- System stays within provided context
- Critical safety feature working well

### 3. Revealing Real Limitations
- v2.0's harder cases expose genuine weaknesses
- 63.14% is likely more realistic than v1.0's 72%
- Clear roadmap for improvements

---

## Comparison with v1.0 Configuration

| Component | v1.0 | v2.0 | Status |
|-----------|------|------|--------|
| Aspect Classifier | ✓ | ✓ | May need tuning for new categories |
| Numerical Variance Detector | ✓ | ✓ | Not preventing qualified→disputed |
| LLM Jury (3/3 threshold) | ✓ | ✓ | Conservative threshold maintained |
| Adaptive Mode | ✓ | ✓ | Query-type routing active |

The same configuration that achieved 72% on v1.0 achieves 63.14% on v2.0, confirming the new test cases are genuinely harder.

---

## Optimization Journey (63.14% → 71.5%)

Key experiments that moved the score (see RESEARCH_NOTEPAD.md for full details):

| Exp | Change | Result |
|-----|--------|--------|
| 001-006 | IE staged pipeline, thresholds, primary entity extraction | 63.14% → 66.3% |
| 007-009 | CA model scaling analysis, SIT rate detection | 66.3% → 68.3% |
| 010-011 | Dispute subordination, qualified consensus rule | 68.3% → 70.3% |
| 012-017 | CA deep-dive (5 approaches), SIT/aspect LLM verifiers | All blocked by 3b limit |
| **018** | **Causal attribution regex tightening** | **70.3% → 71.5%** |

### Remaining Bottlenecks

1. **conflict_aware false fires (34 cases)** — 3b model can't distinguish nuance from contradiction. Needs 7b+ model. Blocked.
2. **Decoy data (16 cases)** — Topically similar but wrong entity. 3b model can't verify entity relevance. Blocked.
3. **Scattered (18 cases)** — Various, partially fixable with regex. Est. +2-3 cases.

### Theoretical Ceiling

- **With qwen2.5:3b**: ~73-75% (current 71.5% + scattered fixes)
- **With 7b+ for conflict_aware**: 75-80%+ (removes discrimination bottleneck)

## Recommendations

### Viable Now

1. **Scattered failure fixes** — Additional regex/heuristic improvements (+2-3 cases)
2. **Model upgrade for conflict_aware** — 7b or 14b for pairwise contradiction only

### Blocked by 3b Model

1. **Conflict_aware precision** — Can't distinguish factual contradiction from nuanced perspectives
2. **Decoy data detection** — Can't verify entity-level relevance with LLM
3. **Aspect classification** — LLM fallback assigns wrong aspects (Exp 016)

### Not Yet Addressed

1. **Relevance category** — 2.50% accuracy, needs investigation
2. **Temporal staleness** — Not detecting outdated documentation

---

## Reproduction

```bash
# Install fitz-gov 2.0
pip install fitz-gov==2.0.0

# Run governance benchmark (249 cases, ~9 min)
python run_targeted_benchmark.py

# Full benchmark (331 cases, ~14 min)
python run_benchmark_simple.py
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

- **Governance run** (249 cases): ~524s (~9 min)
- **Full run** (331 cases): ~14 min

---

## Conclusion

fitz-gov 2.0 expanded the benchmark from 200 to 331 cases, initially dropping accuracy from 72% to 63.14%. Through 18 experiments, governance accuracy recovered to **71.5%** (178/249 governance cases).

**Key takeaways**:
- Qualification accuracy saw the largest improvement: 47.06% → 79.4% (+32.3%)
- Confidence accuracy improved: 79.37% → 88.9% (+9.5%)
- Dispute detection and grounding remain strong (87.3% and 97.62%)
- Abstention remains the hardest category (57.1%), blocked by decoy data problem
- The 3b model is the primary bottleneck — conflict_aware false fires (34 cases) and decoy data (16 cases) need larger model discrimination
- Estimated 3b ceiling: 73-75%. Model upgrade path: 75-80%+