# fitz-gov 1.0: Ablation Studies & Technical Deep-Dive

**Purpose**: Detailed technical analysis of optimization approaches for fitz-gov 1.0 benchmark
**Related**: [fitz-gov-1.0-results.md](fitz-gov-1.0-results.md) for production results

---

## Ablation Study Overview

We tested 12 approaches to optimize governance classification, improving from 49% to 72% accuracy.

### Complete Results Table

| # | Approach | Overall | Abstain | Dispute | Qualify | Confident | Key Finding |
|---|----------|---------|---------|---------|---------|-----------|-------------|
| 1 | Per-chunk stance | 49% | 42.5% | 30% | 47.5% | 100%* | Baseline |
| 2 | Enrichment relevance | 44% | 57.5% | 2.5% | 47.5% | 100% | Summaries hurt disputes |
| 3 | Pairwise detection | 57.5% | 42.5% | 97.5% | 20% | 90% | Dramatic dispute improvement |
| 4 | Model comparison | — | — | — | — | — | Smaller models more decisive |
| 5 | Deterministic | 42.5% | 2.5% | 37.5% | 40% | 96.67% | Antonyms catch ~40% |
| 6 | LLM fusion | 58.5% | 42.5% | 72.5% | 45% | 96.67% | Reduces false positives |
| 7 | Adaptive | 62.5% | 42.5% | 95% | 45% | 93.33% | Best trade-off |
| 8 | + Optimizations | 70.5% | 55% | 95% | 77.5% | 86.67% | Entity matching helps |
| 9 | + Aspect classifier | 72.5% | 72.5% | 92.5% | 72.5% | 86.67% | Major abstention gain |
| 10 | + Numerical filter | 73% | 72.5% | 90% | 77.5% | 86.67% | Handles variations |
| 11 | + Answer verification | 72% | 72.5% | 90% | 72.5% | 86.67% | Epistemic safety |
| 12 | Enhanced enrichment | FAILED | — | — | — | — | Regressions |

*Default fallback, not meaningful

---

## Approach Details

### Approach 1: Per-Chunk Stance Detection (Baseline)

**Method**: Ask LLM about each chunk independently: "Does this answer YES or NO?"

**Implementation**:
```python
for chunk in chunks:
    stance = llm.classify(query, chunk)
    if stance == "YES" or stance == "NO":
        stances.append(stance)
```

**Results**: 49% overall accuracy
- Poor abstention (42.5%) — can't detect irrelevance well
- Poor dispute (30%) — doesn't compare chunks
- Confidence at 100% because it's the default fallback

**Lesson**: Independent chunk analysis insufficient for governance.

---

### Approach 2: Enrichment-Based Relevance

**Method**: Use pre-computed chunk summaries and entities for relevance checking.

**Implementation**:
```python
if chunk.metadata.get("summary"):
    if similarity(query, summary) < 0.45:
        return ABSTAIN
```

**Results**: 44% overall (-5% from baseline!)
- Abstention improved to 57.5% (+15%)
- Dispute **collapsed** to 2.5% (-27.5%)

**Lesson**: Summaries help relevance but destroy contradiction detection because they smooth over conflicting details.

---

### Approach 3: Pairwise Contradiction Detection

**Method**: Show LLM both chunks together for comparison.

**Implementation**:
```python
prompt = f"""
Chunk A: {chunk_a.text}
Chunk B: {chunk_b.text}

Do these chunks CONTRADICT each other? Answer YES or NO.
"""
```

**Results**: 57.5% overall (+8.5%)
- Dispute detection **skyrocketed** to 97.5% (+67.5%)
- Qualification dropped to 20% (-27.5%)

**Lesson**: Pairwise comparison is essential for contradiction detection but over-triggers.

---

### Approach 4: Model Size Comparison

**Method**: Test same pairwise approach across model sizes.

| Model | Overall | Dispute | Qualification | Behavior |
|-------|---------|---------|---------------|----------|
| qwen2.5:3b | 57.5% | 97.5% | 20% | Decisive, binary |
| qwen2.5:7b | 62% | 85% | 47.5% | More nuanced |
| qwen2.5:14b | — | — | — | Too slow for benchmark |

**Lesson**: Smaller models are more aggressive; larger models hedge more naturally.

---

### Approach 5: Deterministic Constraints

**Method**: No LLM calls — embeddings + regex patterns only.

**Components**:
- Embedding similarity for relevance
- Regex antonym detection ("increased/decreased", "up/down")
- Year mismatch detection

**Results**: 42.5% overall
- Abstention nearly zero (2.5%) — embeddings too permissive
- Dispute at 37.5% — antonyms catch obvious contradictions
- No uncertainty detection possible without LLM

**Lesson**: Pure heuristics insufficient; LLM reasoning needed.

---

### Approach 6: LLM Fusion (3-Prompt Voting)

**Method**: Ask contradiction question 3 different ways, require consensus.

**Prompts**:
1. Direct: "Do these CONTRADICT?"
2. Inverted: "Are these CONSISTENT?" (NO = contradict)
3. Logical: "If A is true, can B be true?" (NO = contradict)

**Voting**: 2+ votes required to trigger DISPUTED

**Results**: 58.5% overall
- Dispute down to 72.5% (-25% from pairwise)
- Qualification up to 45% (+25%)
- Fewer false positives

**Lesson**: Fusion reduces noise but can miss true contradictions.

---

### Approach 7: Adaptive Detection

**Method**: Query type determines detection strategy.

**Logic**:
```python
if is_factual_query(query):
    use_pairwise_detection()  # Aggressive
elif is_uncertainty_query(query):
    use_fusion_voting()  # Conservative
```

**Results**: 62.5% overall
- Dispute at 95% (near-optimal)
- Qualification at 45% (improved)
- Best balance achieved

**Lesson**: Different query types need different detection strategies.

---

### Approach 8: Critical Optimizations

**Added**:
1. **Entity matching**: Years and qualifiers must match exactly
2. **Comparative patterns**: Detect "Is X better than Y"
3. **Uncertainty patterns**: Expanded prediction/opinion detection

**Code example**:
```python
# Critical entity extraction
entities = extract_entities(query)  # ["2024", "type 2", "Apple Inc"]
if not any(entity in chunk.text for entity in entities):
    return ABSTAIN
```

**Results**: 70.5% overall (+8% jump)
- Abstention improved to 55%
- Qualification improved to 77.5%

**Lesson**: Domain-specific patterns matter significantly.

---

### Approach 9: Aspect-Aware Entity Matching

**Problem**: Entity matches but different aspect.
- Query: "What causes Alzheimer's?"
- Context: "Alzheimer's symptoms include..."

**Solution**: Classify aspects and check compatibility.

**Aspects**:
- CAUSE, SYMPTOM, TREATMENT
- PRICING, TIMELINE, PROCESS
- DEFINITION, COMPARISON

**Implementation**:
```python
query_aspect = AspectClassifier.classify(query)
chunk_aspects = [AspectClassifier.classify(c) for c in chunks]

if entity_matches but aspect_incompatible:
    return ABSTAIN
```

**Results**: 72.5% overall
- Abstention **jumped** from 55% to 72.5% (+17.5%)
- Minor regressions elsewhere (more conservative)

**Lesson**: Aspect matching crucial for same-entity-different-topic cases.

---

### Approach 10: Numerical Variance Detection

**Problem**: Statistical variations flagged as contradictions.
- "Sales grew 10%"
- "Sales increased 12%"
- Expected: QUALIFIED (both show growth)
- Was getting: DISPUTED

**Solution**: Pre-filter numerical variations.

**Logic**:
```python
def is_variance(num1, num2, unit1, unit2, dir1, dir2):
    if unit1 != unit2:
        return False
    if dir1 != dir2:  # opposite directions = real contradiction
        return False
    if relative_difference(num1, num2) <= 0.25:  # 25% threshold
        return True  # It's variance, not contradiction
```

**Results**: 73% overall
- Qualification improved to 77.5%
- Qualified→disputed errors reduced from 9 to 6

**Lesson**: Domain-aware filtering before LLM checks improves precision.

---

### Approach 11: Answer Verification with LLM Jury

**Problem**: CONFIDENT is default when no constraint triggers, but context might not actually answer the query.

**Solution**: 3-prompt jury to verify context answers query.

**Jury prompts**:
1. "Can this question be answered using the context?"
2. "Is the context INSUFFICIENT to answer?"
3. "Could someone write a complete answer using only this context?"

**Threshold experiments**:

| Threshold | Config | Confidence Accuracy | Decision |
|-----------|--------|---------------------|----------|
| Single call | 1 NO | 26-40% | Too aggressive |
| Majority | 2+ NO | 53.33% | Still too aggressive |
| **Unanimous** | **3/3 NO** | **86.67%** | **Chosen** |

**Results**: 72% overall (slight regression but worth it)
- Confidence preserved at 86.67%
- Added epistemic safety without metric chasing

**Lesson**: Unanimous jury requirement prevents over-correction.

---

### Approach 12: Enhanced Enrichment (FAILED)

**Attempt 1**: Expand summary overlap range
- Changed threshold from 0.45-0.70 to 0.45-0.80
- Result: Confidence dropped 86.67% → 83.33%
- Cause: Too many false abstentions

**Attempt 2**: Relevance verification jury
- 3-prompt jury for borderline cases (0.50-0.75 similarity)
- Problem: Couldn't access similarity scores from constraint
- Result: Ran on all cases, too aggressive

**Conclusion**: Remaining 27.5% abstention failures are genuinely hard. Simple heuristics don't help.

**Lesson**: Not every problem yields to more complexity.

---

## Failed Experiments Archive

### Word Overlap Heuristic
**Tried**: Check if answer words appear in context
**Result**: Confidence 86.67% → 46.67%
**Why**: Can't handle semantic equivalence

### Single-Pass Enrichment
**Tried**: Use summaries for both relevance AND dispute
**Result**: Dispute detection collapsed to 2.5%
**Why**: Summaries smooth over contradictions

### Threshold Tuning
**Tried**: Various similarity thresholds (0.3, 0.5, 0.7, 0.85)
**Result**: No sweet spot — always trade-offs
**Why**: Different query types need different thresholds

---

## Key Lessons Learned

1. **Pairwise comparison essential** — Chunks must be compared together for disputes
2. **Adaptive strategies work** — Different query types need different approaches
3. **Aspect matching crucial** — Entity match alone insufficient
4. **Numerical awareness needed** — Statistical variations ≠ contradictions
5. **Jury voting reduces variance** — Multiple prompts more reliable than one
6. **Conservative thresholds safer** — Unanimous jury prevents over-correction
7. **Not everything needs LLM** — Deterministic pre-filters improve precision
8. **Summaries are double-edged** — Help relevance, hurt contradiction detection
9. **Small models can be better** — More decisive for binary classifications
10. **Perfect is the enemy of good** — 72% with explainable behavior > chasing 80%

---

## Implementation Evolution

### Timeline

- **Feb 2**: Baseline (49%) → Pairwise (57.5%)
- **Feb 2**: Adaptive mode (62.5%) → + Optimizations (70.5%)
- **Feb 3**: Enrichment default, CausalAttribution fix (70%)
- **Feb 4**: Aspect classifier (72.5%)
- **Feb 4**: Numerical variance detector (73%)
- **Feb 4**: Answer verification jury (72%, kept for safety)
- **Feb 4**: Enhanced enrichment experiments (FAILED, reverted)

### Code Impact

| Component | Lines Added | Complexity | Value |
|-----------|-------------|------------|-------|
| Aspect Classifier | ~300 | Medium | +17.5% abstention |
| Numerical Detector | ~200 | Low | +5% qualification |
| LLM Jury | ~150 | Low | Epistemic safety |
| Adaptive Mode | ~100 | Low | Optimal trade-offs |

---

## See Also

- [fitz-gov-1.0-results.md](fitz-gov-1.0-results.md) — Production results and configuration
- [archive/fitz-gov-1.0-tuning.md](archive/fitz-gov-1.0-tuning.md) — Original detailed notes
- [fitz-gov repository](https://github.com/yafitzdev/fitz-gov) — Benchmark implementation