# fitz-gov 2.0: Technical Analysis & Failure Investigation

**Purpose**: Deep technical analysis of v2.0 benchmark failures and improvement strategies
**Related**: [fitz-gov-2.0-results.md](fitz-gov-2.0-results.md) for results summary

---

## Critical Failure: Relevance (2.50%)

### The Problem

Only 1 out of 40 relevance test cases passed. This is a catastrophic failure indicating a fundamental issue.

### Hypothesis

The relevance category tests whether the system answers the actual question asked, not just provides related information. The 97.5% failure rate suggests the system is:
1. Finding tangentially related content
2. Providing information that seems relevant but doesn't address the specific query
3. Not properly aligning answer with question intent

### Investigation Needed

```python
# Load relevance failures for analysis
relevance_cases = [c for c in cases if c.category == FitzGovCategory.RELEVANCE]
failed_cases = benchmark.evaluate(engine, test_cases=relevance_cases)
# Examine specific failure patterns
```

### Potential Root Causes

1. **Answer Verification Constraint Not Triggering**
   - The 3/3 unanimous jury may be too conservative
   - Relevance cases might need different detection

2. **Enrichment Interference**
   - Summaries making irrelevant content seem relevant
   - Entity matching creating false positives

3. **New Subcategories Not Handled**
   - `format_mismatch`: Answer in wrong format
   - `granularity_mismatch`: Answer at wrong level of detail
   - `partial_answer`: Incomplete response marked as complete
   - `wrong_entity_focus`: Discussing related but different entity

---

## Major Regression: Qualification (47.06%)

### The Problem

25.44% drop from v1.0 (72.5% → 47.06%). Main failure: Qualified→Disputed (26 cases, 38.2%).

### New Subcategories Causing Issues

| Subcategory | Description | Likely Failure Mode |
|-------------|-------------|---------------------|
| `entity_ambiguity` | "Apple" - company or fruit? | Should qualify, marks as disputed |
| `scope_ambiguity` | "The project" with multiple projects | Over-confident or false dispute |
| `temporal_ambiguity` | "Current" with multiple time contexts | Picks one instead of qualifying |
| `metric_ambiguity` | "Performance" - speed or quality? | Sees different metrics as contradiction |

### Why Qualified→Disputed?

The system is interpreting different perspectives or aspects as contradictions:

```python
# Example pattern
Query: "How is the product performing?"
Context 1: "Sales are up 20%"        # Financial performance
Context 2: "Load time increased 15%" # Technical performance
Expected: QUALIFIED (different aspects of performance)
Actual: DISPUTED (seen as contradictory claims)
```

### Solution Approaches

1. **Aspect-Aware Contradiction Detection**
   - Extend AspectClassifier to handle ambiguity types
   - Only flag disputes when same aspect conflicts

2. **Ambiguity Detection Module**
   - Detect queries with multiple valid interpretations
   - Force qualification when ambiguity detected

3. **Softer Dispute Thresholds**
   - Require stronger evidence for disputes in ambiguous contexts
   - Use 3/3 jury for disputes on ambiguous queries

---

## Significant Drop: Abstention (57.14%)

### The Problem

15.36% drop from v1.0 (72.5% → 57.14%). Main failure: Abstain→Confident (19 cases, 30.2%).

### New Categories Causing Issues

| Category | Problem | Example |
|----------|---------|---------|
| `code_abstention` | API docs look relevant due to technical terms | Query about Python, gets JavaScript |
| `table_absence` | Answers about missing columns | Query for column Z in table with only A,B,C |
| `temporal_staleness` | Old docs treated as current | 2018 docs for "current" query in 2026 |
| `jurisdictional_mismatch` | Wrong region's laws | US law for EU query |
| `vague_entity_reference` | Ambiguous "the company" | Multiple companies in context |

### Detection Gaps

1. **Temporal Markers Missing**
   ```python
   # Needs detection for:
   - Copyright years
   - "As of [date]" patterns
   - Version numbers
   - "Current" vs actual dates
   ```

2. **Domain/Language Confusion**
   ```python
   # Programming languages mixing:
   - Python query → JavaScript context
   - React query → Vue context
   # Should detect language mismatch
   ```

3. **Structured Data Handling**
   ```python
   # Table/JSON queries need:
   - Schema validation
   - Column existence checking
   - Null/missing value awareness
   ```

---

## Performance by New Subcategory Type

### Code Context Cases

Expected impact on categories:
- **Abstention**: Should detect wrong language/version
- **Qualification**: Should hedge deprecated APIs
- **Confidence**: Clear API docs should be confident
- **Grounding**: Shouldn't hallucinate parameters

Actual performance likely shows:
- Poor abstention (can't detect language mismatch)
- Over-confidence on deprecated APIs
- Good grounding (97.62% overall)

### Ambiguous Query Cases

Expected impact:
- **Qualification**: Should hedge all ambiguous cases
- **Dispute**: Shouldn't see different interpretations as contradictions

Actual performance shows:
- Massive qualification failure (47.06%)
- Qualified→Disputed misclassification

### Structured Data Cases

Expected impact:
- **Confidence**: Clear table/JSON extractions
- **Abstention**: Missing columns/fields
- **Grounding**: No invented data

Mixed results likely:
- Good grounding (no hallucination)
- Poor abstention (answers about missing data)
- Reasonable confidence on clear extractions

---

## Component-Level Diagnosis

### What's Working

| Component | Evidence | Status |
|-----------|----------|--------|
| ConflictAware | 89.09% dispute accuracy | ✅ Robust |
| Grounding constraints | 97.62% grounding | ✅ Excellent |
| Pairwise detection | Stable dispute performance | ✅ Solid |

### What's Failing

| Component | Evidence | Status |
|-----------|----------|--------|
| Relevance detection | 2.50% accuracy | ❌ Broken |
| Qualification triggers | 47.06% accuracy | ❌ Under-triggering |
| Temporal awareness | Abstention failures | ❌ Missing |
| Ambiguity handling | Qualified→Disputed | ❌ Not detecting |

### What's Degraded

| Component | Evidence | Status |
|-----------|----------|--------|
| AspectClassifier | 57.14% abstention | ⚠️ Needs expansion |
| Entity matching | Abstain→Confident | ⚠️ Too permissive |
| Uncertainty detection | Lower qualification | ⚠️ Missing new patterns |

---

## Proposed Fixes

### Priority 1: Fix Relevance (2.50% → Target 70%)

```python
class RelevanceAlignmentConstraint(Constraint):
    """Check if answer addresses the specific question."""

    def check(self, query: str, answer: str) -> bool:
        # 1. Extract question type (what/why/how/when)
        # 2. Extract answer focus
        # 3. Check alignment
        # 4. Use LLM jury if uncertain
```

### Priority 2: Fix Qualification (47.06% → Target 70%)

```python
class AmbiguityDetector:
    """Detect queries with multiple interpretations."""

    def detect_ambiguity(self, query: str, contexts: list) -> AmbiguityType:
        # Check for:
        # - Ambiguous entities
        # - Multiple valid scopes
        # - Temporal ambiguity
        # - Metric ambiguity
```

### Priority 3: Improve Abstention (57.14% → Target 70%)

```python
class EnhancedAbstentionDetector:
    """Temporal, domain, and structural awareness."""

    def should_abstain(self, query: str, context: str) -> bool:
        if self.is_temporally_stale(query, context):
            return True
        if self.is_wrong_domain(query, context):
            return True
        if self.is_missing_structure(query, context):
            return True
```

---

## Testing Strategy

### 1. Isolate New Categories

Test performance on v2.0-specific additions:
```python
# Test only new subcategories
new_cases = [c for c in cases if c.subcategory in NEW_V2_SUBCATEGORIES]
results = benchmark.evaluate(engine, test_cases=new_cases)
```

### 2. Category Ablation

Run without new categories to verify v1.0 performance:
```python
# Test v1.0 subcategories on v2.0 data
v1_style_cases = [c for c in cases if c.subcategory in V1_SUBCATEGORIES]
results = benchmark.evaluate(engine, test_cases=v1_style_cases)
```

### 3. Component Testing

Test individual components:
```python
# Test relevance in isolation
# Test qualification without dispute
# Test abstention without confidence fallback
```

---

## Next Steps

### Immediate (Fix Relevance)
1. Analyze the 39 failed relevance cases
2. Identify common patterns
3. Implement RelevanceAlignmentConstraint
4. Test on v2.0 relevance subset

### Short-term (Fix Qualification)
1. Implement AmbiguityDetector
2. Adjust dispute thresholds for ambiguous queries
3. Test on ambiguous_query subcategories

### Medium-term (Fix Abstention)
1. Add temporal staleness detection
2. Implement domain/language checking
3. Add structured data schema validation
4. Test on new abstention subcategories

---

## Expected Impact

With proposed fixes:

| Category | Current | Target | Improvement Needed |
|----------|---------|--------|-------------------|
| Overall | 63.14% | 72% | +8.86% |
| Relevance | 2.50% | 70% | +67.5% (critical) |
| Qualification | 47.06% | 70% | +22.94% |
| Abstention | 57.14% | 70% | +12.86% |
| Dispute | 89.09% | 90% | Maintain |
| Confidence | 79.37% | 80% | Maintain |
| Grounding | 97.62% | 98% | Maintain |

Fixing relevance alone could improve overall score by ~8% (40 cases × 67.5% improvement ÷ 331 total).