# fitz-gov 2.0 Benchmark Results

**Version**: fitz-gov 2.0.0 (331 test cases)
**Date**: February 6, 2026
**Model**: Ollama qwen2.5:3b (local)
**Runtime**: ~14 minutes

---

## Executive Summary

fitz-gov 2.0 expands the benchmark from 220 to 331 test cases, adding challenging categories for code context, ambiguous queries, structured data, and edge cases. These additions reveal system limitations not captured in v1.0.

### Key Results

| Metric | Score | Assessment |
|--------|-------|------------|
| **Overall Accuracy** | **63.14%** | Significant drop from v1.0, revealing harder cases |
| **Abstention** | 57.14% | Struggles with new code/structured data contexts |
| **Dispute Detection** | 89.09% | Remains robust despite harder cases |
| **Qualification** | 47.06% | Major challenge area with ambiguous queries |
| **Confidence** | 79.37% | Reasonable precision maintained |
| **Grounding** | 97.62% | Excellent hallucination prevention |
| **Relevance** | 2.50% | Critical failure point needing investigation |

### Version Comparison

| Metric | v1.0 (200 cases) | v2.0 (331 cases) | Δ |
|--------|------------------|------------------|-----|
| Overall | 72.0% | 63.14% | -8.86% |
| Abstention | 72.5% | 57.14% | -15.36% |
| Dispute | 90.0% | 89.09% | -0.91% |
| Qualification | 72.5% | 47.06% | -25.44% |
| Confidence | 86.67% | 79.37% | -7.30% |

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

### Primary Failure Modes

| Failure Pattern | Count | Impact |
|-----------------|-------|--------|
| **Qualified→Disputed** | 26 | Over-detecting contradictions in nuanced content |
| **Abstain→Confident** | 19 | Answering despite insufficient context |
| **Abstain→Disputed** | 8 | Finding contradictions in irrelevant content |
| **Qualified→Confident** | 7 | Missing uncertainty that requires hedging |

### Performance by Category

#### Governance Modes

| Category | Accuracy | Correct/Total | Main Failure |
|----------|----------|---------------|--------------|
| Abstention | 57.14% | 36/63 | → Confident (30.2%) |
| Dispute | 89.09% | 49/55 | Stable performance |
| Qualification | 47.06% | 32/68 | → Disputed (38.2%) |
| Confidence | 79.37% | 50/63 | → Disputed (9.5%) |

#### Answer Quality

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

## Recommendations

### Immediate Actions

1. **Investigate Relevance Failure**
   - 2.50% accuracy is catastrophic
   - Check if answer verification constraint is working
   - May need dedicated relevance-checking logic

2. **Tune Qualification Detection**
   - Reduce false disputes on ambiguous content
   - Consider different thresholds for new subcategories
   - May need ambiguity-aware classification

3. **Enhance Temporal Awareness**
   - Detect stale documentation
   - Check temporal markers in context
   - Add recency validation for "current" queries

### Longer-term Improvements

1. **Category-Specific Strategies**
   - Code context needs different handling
   - Structured data requires format-aware logic
   - Ambiguous queries need explicit multi-interpretation support

2. **Relevance Overhaul**
   - Current approach clearly broken on v2.0
   - Consider query-answer alignment scoring
   - May need more sophisticated semantic matching

---

## Reproduction

```bash
# Install fitz-gov 2.0
pip install fitz-gov==2.0.0

# Run benchmark
cd ~/PycharmProjects/fitz-ai
python run_benchmark_simple.py

# Or via CLI (when fixed)
fitz eval fitz-gov --model ollama/qwen2.5:3b
```

### Configuration Used

```python
benchmark = FitzGovBenchmark(
    model_override='ollama/qwen2.5:3b',
    adaptive=True,           # Query-type routing
    use_fusion=True,         # 3-prompt voting
    llm_validation=False     # Disabled for speed
)
```

### Timing

- **Total runtime**: ~14 minutes
- **Per-category breakdown** (from partial run):
  - Abstention (63 cases): 406.9s
  - Dispute (55 cases): 164.4s
  - Qualification (68 cases): 253.5s
  - Confidence (63 cases): 131.0s
  - Grounding (42 cases): 69.3s
  - Relevance (40 cases): ~60s (estimated)

---

## Conclusion

fitz-gov 2.0 successfully expands the benchmark to cover real-world RAG failure modes not captured in v1.0. The 8.86% accuracy drop isn't a regression—it's revealing actual system limitations that need addressing.

**Key takeaways**:
- The benchmark is working as intended: discriminating system capabilities
- Clear problem areas identified: relevance, qualification, abstention
- Dispute detection and grounding remain strong
- The path forward is clear: address relevance first, then qualification handling

The 63.14% score on 331 diverse test cases represents a more realistic assessment of governance capabilities than v1.0's 72% on easier cases.