# FITZ-GOV: Governance Calibration Benchmark

Technical specification and evaluation results for Fitz governance mode classification.

---

## Benchmark Definition

**FITZ-GOV** evaluates **epistemic governance**: whether a RAG system selects an appropriate answer posture (confident, qualified, disputed, abstain) given a query and fixed evidence.

This is not standard accuracy measurement. Standard RAG benchmarks test "did you find the right documents?" FITZ-GOV tests "do you know when you don't know?"

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

FITZ-GOV does **not** evaluate:

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
│ ConflictAware           │ → DISPUTED if chunks contradict
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

---

## Final Results

```
FITZ-GOV Results (n=200):
  Overall Accuracy: 70.50%

Governance Mode Categories:
  abstention: 55.00% (22/40)
  dispute: 95.00% (38/40)
  qualification: 77.50% (31/40)
  confidence: 86.67% (26/30)

Confusion Matrix (rows=expected, cols=actual):
              abstain   disputed   qualifie   confiden
   abstain         22          3          0         15
  disputed          1         38          1          0
 qualified          1          7         31          1
 confident          0          2          2         26
```

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

### 1. Same Entity, Different Aspect

**Rate:** ~45% of abstention failures

**Example:**
- Query: "What causes Alzheimer's disease?"
- Context: "Alzheimer's symptoms include memory loss..."
- Expected: ABSTAIN (context has symptoms, not causes)
- Actual: CONFIDENT (entity "Alzheimer's" matches)

**Root cause:** Entity matching confirms topic, but doesn't verify aspect (causes vs symptoms vs treatment).

### 2. Semantic Similarity Without Lexical Match

**Rate:** ~30% of abstention failures

**Example:**
- Query: "How do I use Python for web scraping?"
- Context: "Python is widely used for data science..."
- Expected: ABSTAIN (different use case)
- Actual: CONFIDENT (high embedding similarity, entity "Python" matches)

**Root cause:** Same technology, different application. Embeddings see similarity; aspect differs.

### 3. Qualification → Disputed Misclassification

**Rate:** 7/40 qualification cases (17.5%)

**Example:**
- Query: "Why did website traffic decrease?"
- Context: Contains traffic statistics
- Expected: QUALIFIED (causal query with only correlational data)
- Actual: DISPUTED (conflict detection triggers on statistical variation)

**Root cause:** Statistical variations mistaken for contradictions.

### 4. No Extractable Entities

**Rate:** ~10% of abstention failures

**Example:**
- Query: "How do I fix a leaky faucet?"
- Context: About Shakespeare plays
- Expected: ABSTAIN
- Actual: DISPUTED (similarity just above threshold, no entities to check)

**Root cause:** Query has no proper nouns/years/qualifiers to extract.

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
from fitz_ai.engines.fitz_rag import FitzRagEngine
from fitz_ai.evaluation.benchmarks import FitzGovBenchmark

config = load_engine_config('fitz_rag')
engine = FitzRagEngine(config)
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
| Constraint runner | `fitz_ai/core/guardrails/runner.py` |

---

## Changelog

- **2026-02-02:** Initial benchmark at 49%. Pairwise detection → 57.5%. Adaptive mode → 62.5%. Entity matching + uncertainty patterns → 70.5%.

---

## References

- [Detailed methodology and narrative](../blog/governance-constraint-experiments.md)
- [Feature documentation](../features/governance-benchmarking.md)
