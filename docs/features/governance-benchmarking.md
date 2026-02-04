# Governance Benchmarking (fitz-gov)

## The Problem with RAG Benchmarks

Standard RAG benchmarks (BEIR, MS MARCO, etc.) measure **retrieval quality**: "Did you find the right documents?"

They don't measure **epistemic honesty**: "Do you know when you don't know?"

A RAG system that hallucinates confidently on every query would score well on retrieval benchmarks—it retrieved the right docs, after all. But it's useless for high-stakes applications where "I don't know" is the correct answer.

## fitz-gov: Governance Calibration Benchmark

fitz-gov tests whether a RAG system correctly classifies its own confidence level across 150+ test cases:

| Mode | What It Tests | Example |
|------|---------------|---------|
| **ABSTAIN** | Context doesn't answer the question | Query: "Tokyo population" / Context: about Osaka |
| **DISPUTED** | Sources contradict each other | "Project succeeded" vs "Project failed" |
| **QUALIFIED** | Evidence exists but needs caveats | Causal question with only correlational data |
| **CONFIDENT** | Clear, consistent evidence | Direct factual answer supported by sources |

Each test case has:
- A query
- Injected context (bypassing retrieval for controlled testing)
- Expected governance mode

## Current Results

**Fitz achieves 72% governance accuracy** with aspect-aware matching, numerical variance detection, and LLM jury verification:

| Category | Accuracy | What It Means |
|----------|----------|---------------|
| **Overall** | **72%** | Correct mode classification |
| Abstention | 72.5% | Correctly refuses when context is irrelevant |
| Dispute | 90% | Detects contradictions between sources |
| Qualification | 72.5% | Adds caveats for uncertain claims |
| Confidence | 86.67% | Confident when evidence is clear |

For comparison, a naive system that always outputs CONFIDENT scores ~20% (only the confidence cases).

**Design philosophy:** We prioritize a defensible system over maximum metrics. The LLM jury adds epistemic safety—when all 3 prompts unanimously agree context doesn't answer, we qualify rather than risk false confidence.

## Running the Benchmark

```python
from fitz_ai.engines.fitz_rag import FitzRagEngine
from fitz_ai.evaluation.benchmarks import FitzGovBenchmark

engine = FitzRagEngine(config)
benchmark = FitzGovBenchmark(adaptive=True)
results = benchmark.evaluate(engine)
print(results)
```

Or via CLI:
```bash
fitz eval fitz-gov --collection test
```

## Key Techniques

### 1. Pairwise Contradiction Detection
Instead of classifying each chunk independently, compare chunks together:
```
"Do these two texts contradict each other about the question?"
```
This improved dispute detection from 30% → 95%.

### 2. Critical Entity Matching
Years and numbered qualifiers (like "type 2" in "type 2 diabetes") must match exactly—high embedding similarity isn't enough:
```
Query: "2024 World Series" + Context: "2021 World Series" → ABSTAIN
Query: "type 2 diabetes" + Context: "type 1 diabetes" → ABSTAIN
```

### 3. Adaptive Detection
Use query type to select detection method:
- **Factual queries** → Aggressive contradiction detection (high recall)
- **Uncertainty queries** ("Why?", "Should we?") → Conservative detection (high precision)

### 4. Expanded Uncertainty Patterns
Not just "why" questions need qualification—also:
- Predictive: "Will X happen next year?"
- Opinion: "Is X better than Y?"
- Speculative: "What percentage will...?"

## Technical Specification

For the authoritative technical record with full ablation results, trade-offs, and known failure modes:

**[docs/evaluation/governance.md](../evaluation/governance.md)** ← Canonical reference

For the narrative exploration of how we arrived at these results:
[Governance Constraint Experiments (blog)](../blog/governance-constraint-experiments.md)

## Files

- **Benchmark:** `fitz_ai/evaluation/benchmarks/fitz_gov.py`
- **Test cases:** [fitz-gov package](https://github.com/yafitzdev/fitz-gov)
- **Constraints:**
  - `fitz_ai/core/guardrails/plugins/insufficient_evidence.py` - Abstention detection
  - `fitz_ai/core/guardrails/plugins/conflict_aware.py` - Contradiction detection
  - `fitz_ai/core/guardrails/plugins/causal_attribution.py` - Uncertainty detection

## Why This Matters

| Domain | Why Governance Accuracy Matters |
|--------|--------------------------------|
| **Legal** | "I don't know" is better than wrong legal advice |
| **Medical** | Qualified answers prevent dangerous overconfidence |
| **Compliance** | Auditors need to know when evidence is missing |
| **Finance** | Disputed data should surface conflicts, not pick arbitrarily |

Standard RAG benchmarks optimize for "find the right document." fitz-gov optimizes for "know what you don't know."

## Related Features

- [Epistemic Honesty](epistemic-honesty.md) - The constraint system that enables governance
- [Hierarchical RAG](hierarchical-rag.md) - Corpus summaries help detect missing information
- [Multi-Hop Reasoning](multi-hop-reasoning.md) - Iterative retrieval reduces false abstentions
