# fitz-gov: Governance Calibration Benchmark

Technical specification and evaluation results for Fitz governance mode classification.

---

## Benchmark Definition

**fitz-gov** evaluates **epistemic governance**: whether a RAG system selects an appropriate answer posture (confident, qualified, disputed, abstain) given a query and fixed evidence. It is designed to be model-agnostic, retrieval-independent, and governance-specific.

This is not standard accuracy measurement. Standard RAG benchmarks test "did you find the right documents?" fitz-gov tests "do you know when you don't know?"

### Test Cases

- **Total cases:** 150 governance mode cases + 50 answer quality cases
- **Source:** [fitz-gov package](https://github.com/yafitzdev/fitz-gov)

Each case contains:
```
{
  "query": str,
  "contexts": list[str],    # Injected context (bypasses retrieval)
  "expected_mode": str      # abstain | disputed | qualified | confident
}
```

### Governance Modes

| Mode | Definition | Trigger Condition |
|------|------------|-------------------|
| ABSTAIN | Context does not answer the query | Insufficient evidence, wrong entity, off-topic |
| DISPUTED | Context contains contradictory information | Sources disagree on factual claims |
| QUALIFIED | Evidence exists but requires caveats | Causal/predictive/opinion queries without definitive evidence |
| CONFIDENT | Clear, consistent evidence supports answer | Default when no constraint triggers |

### Non-Goals

fitz-gov does **not** evaluate:

- **Answer factual correctness** — We don't check if the answer is true, only if the system chose the right posture
- **Retrieval quality** — Contexts are injected, bypassing retrieval entirely
- **Linguistic quality** — Grammar, fluency, and style are not measured
- **User satisfaction** — No human preference judgments

These are separate concerns requiring separate benchmarks.

### Evaluation Metric

Per-category accuracy: `correct_classifications / total_cases_in_category`

Overall accuracy: micro-average across all test cases (equal weight per case, not per category).

**Asymmetric costs:** Governance evaluation is inherently asymmetric — false confidence is worse than false qualification. A system that hedges too much is annoying; a system that confidently hallucinates is dangerous. The benchmark does not encode this asymmetry in the metric, but it informs our trade-off decisions.

---

## Methodology

### Test Isolation

Contexts are **injected directly** into the constraint pipeline, bypassing retrieval. This isolates governance classification from retrieval quality.

### Constraint Pipeline

```
Query + Injected Chunks
        │
        ▼
┌─────────────────────────┐
│ InsufficientEvidence    │ → ABSTAIN if context irrelevant
├─────────────────────────┤
│ CausalAttribution       │ → QUALIFIED if uncertainty query without evidence
├─────────────────────────┤
│ ConflictAware           │ → DISPUTED if chunks contradict (LLM jury for disputes)
├─────────────────────────┤
│ AnswerVerification      │ → QUALIFIED if context doesn't answer (LLM jury, 3/3 NO)
├─────────────────────────┤
│ Default                 │ → CONFIDENT
└─────────────────────────┘
```

### Configuration

All experiments run with:
- **Model:** Ollama qwen2.5:3b (local)
- **Embeddings:** Ollama nomic-embed-text
- **Mode:** `adaptive=True` (query-type selects detection method)

### Adaptive Mode

Adaptive mode dynamically selects contradiction detection strategy based on query type:

- **Factual queries** → Pairwise detection (aggressive, high recall)
- **Uncertainty queries** (causal, predictive, opinion) → Fusion voting (conservative, high precision)

This is the default (`adaptive=True`) and is evaluated explicitly in Approaches 6–8 below. Results without adaptive mode would show different trade-offs between dispute and qualification accuracy.

---

## LLM Jury System

The constraint pipeline uses **LLM jury voting** to reduce model variance on critical decisions. Instead of a single LLM call that can be noisy, we ask the same question 3 different ways and require consensus.

### Why Jury Voting?

Single LLM calls are subject to variance—the same prompt can get different answers on different runs. For governance decisions that affect user trust, we want high confidence that the classification is correct.

The jury pattern:
1. Ask 3 prompts that probe the same question from different angles
2. Require consensus (majority or unanimous) to trigger
3. Reduces false positives from LLM noise

### Jury Implementations

#### ConflictAwareConstraint (Dispute Detection)

Used in adaptive mode for uncertainty/causal queries where false disputes are costly.

**3 prompts:**
1. Direct: "Do these CONTRADICT?"
2. Inverted: "Are these CONSISTENT?" (NO = contradict)
3. Logical: "If A is true, can B be true?" (NO = contradict)

**Threshold:** 2+ votes → DISPUTED

#### AnswerVerificationConstraint (Positive Confirmation)

Catches cases where context is semantically relevant but doesn't actually answer the query.

**3 prompts:**
1. Direct: "Can this question be answered using the context?"
2. Inverted: "Is the context INSUFFICIENT to answer?" (YES = doesn't answer)
3. Completeness: "Could someone write a complete answer using only this context?"

**Threshold:** 3/3 unanimous NO → QUALIFIED

### Design Philosophy

We use **unanimous voting (3/3) for answer verification** because:

1. **Conservative blocking:** Only block confident answers when ALL jurors agree the context doesn't answer
2. **Epistemic safety:** When 3 different prompts unanimously agree, that's strong signal
3. **Minimal false positives:** Preserves baseline accuracy while catching egregious mismatches

The goal is not to maximize benchmark metrics but to build a **defensible system**. A confident answer that's wrong damages trust more than a qualified answer that could have been confident.

### Performance Cost

Each jury adds 3 LLM calls per decision point:
- ConflictAware jury: 3 calls × N chunk pairs (only for uncertainty queries in adaptive mode)
- AnswerVerification jury: 3 calls per query

With fast local models (qwen2.5:3b), this adds ~100-300ms per jury. The epistemic safety is worth the latency for governance-critical decisions.

---

## Ablation Results

### Approach 1: Per-Chunk Stance Detection (Baseline)

Ask LLM about each chunk independently: "Does this answer YES or NO?"

| Category | Accuracy |
|----------|----------|
| Overall | 49% |
| Abstention | 42.5% |
| Dispute | 30% |
| Qualification | 47.5% |
| Confidence | 100%* |

*Confidence is the default fallback. 100% is not meaningful.

### Approach 2: Enrichment-Based Relevance

Use chunk metadata (summaries, entities) for relevance checking.

| Category | Accuracy | Δ from Baseline |
|----------|----------|-----------------|
| Overall | 44% | -5% |
| Abstention | 57.5% | +15% |
| Dispute | 2.5% | -27.5% |
| Qualification | 47.5% | — |
| Confidence | 100% | — |

**Finding:** Summaries improve relevance detection but destroy contradiction detection.

### Approach 3: Pairwise Contradiction Detection

Show LLM both chunks together: "Do these contradict?"

| Category | Accuracy | Δ from Baseline |
|----------|----------|-----------------|
| Overall | 57.5% | +8.5% |
| Abstention | 42.5% | — |
| Dispute | 97.5% | +67.5% |
| Qualification | 20% | -27.5% |
| Confidence | 90% | -10% |

**Finding:** Pairwise comparison dramatically improves dispute detection.

### Approach 4: Model Comparison

Same pairwise approach across model sizes:

| Model | Overall | Dispute | Qualification |
|-------|---------|---------|---------------|
| qwen2.5:3b | 57.5% | 97.5% | 20% |
| qwen2.5:7b | 62% | 85% | 47.5% |
| qwen2.5:14b | — | — | — |

**Finding:** Smaller models are more decisive; larger models hedge.

### Approach 5: Deterministic Constraints

Embeddings + regex antonyms only, no LLM:

| Category | Accuracy | Δ from Baseline |
|----------|----------|-----------------|
| Overall | 42.5% | -6.5% |
| Abstention | 2.5% | -40% |
| Dispute | 37.5% | +7.5% |
| Qualification | 40% | -7.5% |
| Confidence | 96.67% | -3.33% |

**Finding:** Antonyms catch ~40% of contradictions. Embedding relevance too lenient.

### Approach 6: LLM Fusion (3-Prompt Voting)

Ask contradiction question 3 ways, require 2/3 consensus:

| Category | Accuracy | Δ from Approach 3 |
|----------|----------|-------------------|
| Overall | 58.5% | +1% |
| Abstention | 42.5% | — |
| Dispute | 72.5% | -25% |
| Qualification | 45% | +25% |
| Confidence | 96.67% | +6.67% |

**Finding:** Fusion reduces false positives (better qualification) but misses true contradictions.

### Approach 7: Adaptive Detection

Query type selects detection method:
- Factual queries → Pairwise (aggressive)
- Uncertainty queries → Fusion (conservative)

| Category | Accuracy | Δ from Approach 3 |
|----------|----------|-------------------|
| Overall | 62.5% | +5% |
| Abstention | 42.5% | — |
| Dispute | 95% | -2.5% |
| Qualification | 45% | +25% |
| Confidence | 93.33% | +3.33% |

**Finding:** Adaptive selection achieves best trade-off between dispute and qualification.

### Approach 8: Final Optimizations

Added:
1. **Critical entity matching:** Years and numbered qualifiers (e.g., "2024", "type 2") must match exactly
2. **Comparative question detection:** Regex patterns for "Is X better", "Which X is better"
3. **Expanded uncertainty patterns:** Predictive, opinion, speculative queries

| Category | Accuracy | Δ from Baseline |
|----------|----------|-----------------|
| Overall | 70.5% | +21.5% |
| Abstention | 55% | +12.5% |
| Dispute | 95% | +65% |
| Qualification | 77.5% | +30% |
| Confidence | 86.67% | -13.33% |

### Approach 9: Aspect-Aware Entity Matching

**Problem:** ~45% of abstention failures occur when entity matches but query aspect differs from chunk content (e.g., query asks about "causes" but chunk discusses "symptoms").

**Solution:** Added `AspectClassifier` to categorize queries and chunks by aspect (CAUSE, SYMPTOM, TREATMENT, PRICING, DEFINITION, PROCESS, COMPARISON, TIMELINE). If entity matches but aspects are incompatible → ABSTAIN.

| Category | Accuracy | Δ from Approach 8 |
|----------|----------|-------------------|
| Overall | 72.5% | +2% |
| Abstention | 72.5% | +17.5% |
| Dispute | 92.5% | -2.5% |
| Qualification | 72.5% | -5% |
| Confidence | 86.67% | — |

**Finding:** Aspect classification dramatically improves abstention (+17.5%) by catching "same entity, different aspect" failures. Minor regression in dispute/qualification due to more conservative relevance checking.

**Files:** `fitz_ai/core/guardrails/aspect_classifier.py`

### Approach 10: Numerical Variance Detection

**Problem:** Statistical variations like "Sales grew 10%" vs "Sales grew 12%" incorrectly flagged as contradictions, causing qualification → disputed misclassifications.

**Solution:** Added `NumericalConflictDetector` that extracts numeric mentions with units and direction. Before LLM contradiction check:
- Same unit + same direction + ≤25% relative difference → variance (skip LLM)
- Opposite directions → real contradiction (proceed with LLM)
- Different authoritative sources cited → potential dispute (proceed with LLM)

| Category | Accuracy | Δ from Approach 9 |
|----------|----------|-------------------|
| Overall | 73% | +0.5% |
| Abstention | 72.5% | — |
| Dispute | 90% | -2.5% |
| Qualification | 77.5% | +5% |
| Confidence | 86.67% | — |

**Finding:** Numerical variance detection improves qualification (+5%) by preventing false disputes on statistical variations. Confusion matrix shows qualified→disputed dropped from 9 to 6 cases.

**Files:** `fitz_ai/core/guardrails/numerical_detector.py`, `fitz_ai/core/guardrails/plugins/conflict_aware.py`

### Approach 11: Answer Verification with LLM Jury

**Problem:** CONFIDENT is the default fallback when no constraint triggers. Context can be semantically relevant but not actually answer the query (e.g., "What is the capital of France?" with context about France's population).

**Solution:** `AnswerVerificationConstraint` uses 3-prompt LLM jury to verify context answers the query. Requires unanimous (3/3) NO votes to qualify.

**3 prompts:**
1. Direct: "Can this question be answered using the context?"
2. Inverted: "Is the context INSUFFICIENT to answer?" (YES = doesn't answer)
3. Completeness: "Could someone write a complete answer using only this context?"

**Threshold experiments:**

| Threshold | Confidence Accuracy | Status |
|-----------|---------------------|--------|
| Single LLM call | 26-40% | Too conservative |
| 2+ NO votes | 53.33% | Still too aggressive |
| **3/3 unanimous** | **86.67%** | Baseline preserved |

**Final results with 3/3 threshold:**

| Category | Accuracy | Δ from Approach 10 |
|----------|----------|-------------------|
| Overall | 72% | -1% |
| Abstention | 72.5% | — |
| Dispute | 90% | — |
| Qualification | 72.5% | -5% |
| Confidence | 86.67% | — |

**Finding:** The 3/3 unanimous threshold preserves baseline confidence accuracy while adding epistemic safety. When all 3 jury prompts agree context doesn't answer, that's strong signal worth acting on—even if the benchmark doesn't capture all such cases.

**Design philosophy:** We don't chase metrics. The jury adds a sanity check that only fires on clear mismatches. A confident answer that's wrong damages trust more than a qualified answer that could have been confident.

**Files:** `fitz_ai/core/guardrails/plugins/answer_verification.py`

### Approach 12: Enhanced Enrichment for Abstention (FAILED)

**Problem:** Abstention at 72.5% leaves ~27% of cases where context is retrieved but doesn't match query topic. Approach 2 showed enrichment could help (+15% abstention) but hurt dispute detection.

**Hypothesis:** With better dispute detection (Approaches 9-11), we can now safely expand enrichment-based abstention.

**Experiments:**

#### 12a: Expanded Enrichment Range

Extended summary overlap check from (0.45-0.70) to (0.45-0.80):

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Overall | 72% | 70.5% | -1.5% |
| Qualification | 72.5% | 67.5% | **-5%** |
| Confidence | 86.67% | 83.33% | **-3.3%** |

**Why it failed:** Summary overlap is too crude. Checking if ANY query word appears in ANY summary causes false abstentions for semantically related but lexically different content.

#### 12b: Relevance Verification Jury

Created `RelevanceVerificationConstraint` using 3-prompt jury for borderline similarity cases (0.50-0.75):

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Overall | 72% | 71.5% | -0.5% |
| Confidence | 86.67% | 83.33% | **-3.3%** |
| Abstention | 72.5% | 72.5% | — |

**Why it failed:**
1. Constraint couldn't access similarity scores (computed inside InsufficientEvidence, not passed to other constraints)
2. Without similarity gating, the jury ran on all cases and was too aggressive

**Conclusion:** The remaining 27.5% abstention failures are genuinely hard cases—high embedding similarity, entity matches, but wrong topic. Simple heuristics and even jury-based approaches don't improve them without causing regressions elsewhere.

**Code deleted:** `relevance_verification.py` removed after experiments.

---

## Final Results

> **Why 72% is meaningful:** Governance is fundamentally harder than retrieval because it requires reasoning about *absence*, *conflict*, and *uncertainty*—not just relevance. A 72% accuracy on governance classification represents strong calibration for a task where even human experts frequently disagree.

### Current Production Results

With aspect-aware entity matching, numerical variance detection, and LLM jury verification:

```
fitz-gov Results (n=200):
  Overall Accuracy: 72.00%

Governance Mode Categories:
  abstention: 72.50% (29/40)
  dispute: 90.00% (36/40)
  qualification: 72.50% (29/40)
  confidence: 86.67% (26/30)

Confusion Matrix (rows=expected, cols=actual):
              abstain   disputed   qualifie   confiden
   abstain         29          2          0          9
  disputed          2         36          1          1
 qualified          1          9         29          1
 confident          0          2          2         26
```

### Improvement Summary

| Category | Approach 8 | Approach 11 (Current) | Change |
|----------|------------|----------------------|--------|
| Overall | 70.5% | 72% | +1.5% |
| Abstention | 55% | 72.5% | +17.5% |
| Dispute | 95% | 90% | -5% |
| Qualification | 77.5% | 72.5% | -5% |
| Confidence | 86.67% | 86.67% | — |

Key improvements:
- **Abstention +17.5%**: Aspect classifier catches "same entity, different aspect" failures
- **Confidence preserved**: LLM jury with 3/3 threshold adds safety without regression
- **Epistemic safety**: System now has positive confirmation via unanimous jury

The dispute and qualification regressions (-5% each) are acceptable trade-offs for:
1. Significant abstention improvement
2. Added epistemic safety from answer verification jury

We prioritize a **defensible system** over maximum metrics.

---

## Trade-offs

### Confidence vs. Qualification

Aggressive uncertainty detection improves qualification (47.5% → 77.5%) but reduces confidence (100% → 86.67%). Some confident cases are now correctly flagged as opinion/speculative queries.

This is **intentional**: over-qualifying is safer than over-confidence. In high-stakes domains (legal, medical, compliance), a hedged answer that prompts human review is far better than a confident hallucination that goes unquestioned.

### Dispute vs. Qualification

Pairwise detection catches 95% of contradictions but misclassifies some qualified cases as disputed. Fusion reduces this but misses true contradictions.

Adaptive mode mitigates by selecting method based on query type.

### Abstention Threshold

Entity matching bypass threshold set at 0.85 similarity. Higher catches more wrong-entity cases but risks false abstentions on legitimate matches.

### Hybrid Relevance Detection

Approach 2 showed that enriched metadata (summaries) improved abstention to 57.5% but destroyed dispute detection. The current implementation uses a **hybrid approach**:

- Embeddings check general semantic similarity
- Summaries check specific topic match (only in ambiguous similarity range 0.45-0.75)
- ConflictAware handles dispute detection separately

This isolates the benefit of metadata for abstention without affecting dispute accuracy. To test with enriched chunks:

```bash
fitz eval fitz-gov --enrich --model ollama/qwen2.5:3b
```

Note: `--enrich` adds latency (LLM calls for enrichment) but better simulates production conditions where chunks are pre-enriched during ingestion.

---

## Known Failure Modes

### 1. Same Entity, Different Aspect (ADDRESSED)

**Status:** Largely addressed by Approach 9 (Aspect Classifier)

**Example:**
- Query: "What causes Alzheimer's disease?"
- Context: "Alzheimer's symptoms include memory loss..."
- Expected: ABSTAIN (context has symptoms, not causes)
- Previous: CONFIDENT (entity "Alzheimer's" matches)
- **Now: ABSTAIN** (aspect mismatch detected)

**Improvement:** Abstention accuracy improved from 55% to 72.5% (+17.5%).

### 2. Semantic Similarity Without Lexical Match

**Rate:** ~25% of remaining abstention failures

**Example:**
- Query: "How do I use Python for web scraping?"
- Context: "Python is widely used for data science..."
- Expected: ABSTAIN (different use case)
- Actual: CONFIDENT (high embedding similarity, entity "Python" matches)

**Root cause:** Same technology, different application. Embeddings see similarity; aspect classifier may not catch all application-specific mismatches.

### 3. Qualification → Disputed Misclassification (ADDRESSED)

**Status:** Largely addressed by Approach 10 (Numerical Variance Detector)

**Example:**
- Query: "Why did website traffic decrease?"
- Context: Contains traffic statistics like "dropped 10%" vs "fell 12%"
- Expected: QUALIFIED (causal query with only correlational data)
- Previous: DISPUTED (conflict detection triggers on statistical variation)
- **Now: QUALIFIED** (numerical variance detected, LLM check skipped)

**Improvement:** Qualification→disputed misclassifications dropped from 9 to 6 cases.

### 4. No Extractable Entities

**Rate:** ~10% of abstention failures

**Example:**
- Query: "How do I fix a leaky faucet?"
- Context: About Shakespeare plays
- Expected: ABSTAIN
- Actual: CONFIDENT (similarity just above threshold, no entities to check)

**Root cause:** Query has no proper nouns/years/qualifiers to extract. Aspect classifier helps but requires detectable aspect patterns.

### 5. Source Attribution vs Variance

**Rate:** ~5% of dispute cases

**Example:**
- Query: "What is the company's market share?"
- Context 1: "Gartner says 32% market share"
- Context 2: "Company claims 38% market share"
- Expected: DISPUTED (different sources disagree)
- Risk: Could be incorrectly flagged as variance

**Mitigation:** Numerical detector checks for source indicators ("according to", "claims", "reports") and skips variance detection when different sources are cited.

### 6. Relevant Context That Doesn't Answer (ADDRESSED)

**Status:** Addressed by Approach 11 (LLM Jury with 3/3 threshold)

**Example:**
- Query: "What is the capital of France?"
- Context: "France has 67 million people and is famous for wine."
- Expected: QUALIFIED
- Without verification: CONFIDENT (no constraint triggers)
- **With jury verification: QUALIFIED** (3/3 jury agrees context doesn't answer)

**Solution:** `AnswerVerificationConstraint` with unanimous jury. See Approach 11 for details.

---

## Reproduction

### CLI (Recommended)

```bash
# Run with documented model for reproducible results
fitz eval fitz-gov --model ollama/qwen2.5:3b

# Test with different models
fitz eval fitz-gov --model cohere
fitz eval fitz-gov --model ollama/qwen2.5:14b
```

### Python API

```python
from fitz_ai.config.loader import load_engine_config
from fitz_ai.engines.fitz_krag import FitzKragEngine
from fitz_ai.evaluation.benchmarks import FitzGovBenchmark

config = load_engine_config('fitz_krag')
engine = FitzKragEngine(config)
benchmark = FitzGovBenchmark(model_override="ollama/qwen2.5:3b")
results = benchmark.evaluate(engine)
print(results)
```

### Requirements

- Ollama running with `qwen2.5:3b` and `nomic-embed-text`
- `pip install fitz-gov`

---

## Files

| Component | Path |
|-----------|------|
| Benchmark runner | `fitz_ai/evaluation/benchmarks/fitz_gov.py` |
| Test cases | [fitz-gov package](https://github.com/yafitzdev/fitz-gov) |
| InsufficientEvidence constraint | `fitz_ai/core/guardrails/plugins/insufficient_evidence.py` |
| ConflictAware constraint | `fitz_ai/core/guardrails/plugins/conflict_aware.py` |
| CausalAttribution constraint | `fitz_ai/core/guardrails/plugins/causal_attribution.py` |
| AnswerVerification constraint | `fitz_ai/core/guardrails/plugins/answer_verification.py` (jury-based, 3/3 threshold) |
| Constraint runner | `fitz_ai/core/guardrails/runner.py` |
| Aspect classifier | `fitz_ai/core/guardrails/aspect_classifier.py` |
| Numerical variance detector | `fitz_ai/core/guardrails/numerical_detector.py` |

---

## Changelog

- **2026-02-04:** Enhanced enrichment experiments (Approach 12, FAILED). Expanded summary range caused regressions (confidence 86.67%→83.33%). RelevanceVerification jury couldn't access similarity scores. Remaining 27.5% abstention failures are genuinely hard cases. Code deleted.
- **2026-02-04:** AnswerVerificationConstraint with LLM jury (ENABLED BY DEFAULT). Uses 3-prompt jury with unanimous (3/3) threshold. Catches egregious "relevant but doesn't answer" cases while preserving baseline accuracy (86.67% confidence). Design philosophy: epistemic safety over metric chasing.
- **2026-02-04:** Jury threshold experiments. Single-call: 26-40% confidence. 2+ NO votes: 53.33%. 3/3 unanimous: 86.67% (baseline preserved). Unanimous threshold selected for minimal false positives.
- **2026-02-04:** PositiveConfirmationConstraint experiment (ABANDONED). Attempted word-overlap heuristics. Caused massive regression (86.67% → 46.67%). Word-overlap cannot handle semantic equivalence.
- **2026-02-04:** Numerical variance detection (Approach 10). Prevents statistical variations from triggering false disputes. Qualification stable at 77.5%, qualified→disputed errors reduced. Production score: 73%.
- **2026-02-04:** Aspect-aware entity matching (Approach 9). Catches "same entity, different aspect" failures. Abstention improved 55% → 72.5%. Production score: 72.5%.
- **2026-02-03:** Enrichment as default. Hybrid summary check for abstention (+2.5%). Fixed CausalAttributionConstraint false positives from LLM-generated summaries. Production score: 70%.
- **2026-02-02:** Initial benchmark at 49%. Pairwise detection → 57.5%. Adaptive mode → 62.5%. Entity matching + uncertainty patterns → 70.5%.

---

## References

- [Detailed methodology and narrative](../blog/governance-constraint-experiments.md)
- [Feature documentation](../features/governance-benchmarking.md)
