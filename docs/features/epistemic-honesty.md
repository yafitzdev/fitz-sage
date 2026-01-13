# Epistemic Honesty

## Problem

Most RAG systems confidently answer questions even when the answer isn't in the documents:

- **Q:** "What was our Q4 revenue?" (docs only cover Q1-Q3)
- **Typical RAG:** "Q4 revenue was $2.5M" (hallucinated)
- **FitzRAG:** "I cannot find Q4 revenue figures in the provided documents. The available financial data covers Q1-Q3 only."

The system cannot distinguish between "I have evidence" and "I'm making an educated guess."

## Solution: Epistemic Guardrails

Fitz has built-in constraint plugins that detect uncertainty and refuse to answer when evidence is insufficient:

```
Q: "What was our Q4 revenue?"
A: "I cannot find Q4 revenue figures in the provided documents.
    The available financial data covers Q1-Q3 only."

   Mode: ABSTAIN
```

## How It Works

### Constraint Plugins

Three guardrails run automatically on every answer:

| Constraint | What It Catches | Example |
|------------|-----------------|---------|
| **ConflictAware** | Sources disagree | "Document A says X, but Document B says Y. This is a conflict." |
| **InsufficientEvidence** | No supporting evidence | "I cannot find information about X in the provided documents." |
| **CausalAttribution** | Correlation ≠ causation | "The data shows a correlation, but I cannot determine causation without additional evidence." |

### Answer Modes

Every answer includes a **mode** indicating confidence level:

- `CONFIDENT` — Strong evidence supports the answer across multiple sources
- `QUALIFIED` — Answer given with noted limitations or caveats
- `DISPUTED` — Sources conflict; both views are presented
- `ABSTAIN` — Insufficient evidence; refuses to answer

## Key Design Decisions

1. **Always-on** - Constraints run automatically on every answer. No configuration needed.

2. **Post-generation filtering** - Constraints evaluate the LLM's answer and retrieved chunks, not the raw query.

3. **Explicit modes** - The mode field is first-class in the Answer dataclass, not a hidden flag.

4. **Fail-safe defaults** - When in doubt, ABSTAIN. Better to say "I don't know" than to hallucinate.

5. **Transparent reasoning** - When abstaining or qualifying, the system explains why.

## Configuration

No configuration required. Constraints are baked into the answer generation pipeline.

To disable (not recommended):
```yaml
# config.yaml
constraints:
  enabled: false
```

## Files

- **Constraint plugins:** `fitz_ai/core/guardrails/plugins/`
  - `conflict_aware.py` - Detects contradictions across sources
  - `insufficient_evidence.py` - Blocks confident answers without evidence
  - `causal_attribution.py` - Prevents hallucinated causality
- **Constraint runner:** `fitz_ai/engines/fitz_rag/answering/constraints.py`
- **Answer modes:** `fitz_ai/core/answer.py` (AnswerMode enum)

## Benefits

| Without Epistemic Honesty | With Epistemic Honesty |
|---------------------------|------------------------|
| Hallucinated answers look confident | "I don't know" when uncertain |
| No way to detect conflicts | Surfaces contradictions explicitly |
| Users can't trust output | Transparent confidence signaling |
| Dangerous for high-stakes domains | Safe for compliance, legal, medical |

## Example

**Query:** "What caused the Q4 sales decline?"

**Without constraints:**
```
Answer: The Q4 sales decline was primarily caused by increased competition
and seasonal factors.

Mode: CONFIDENT
```

**With constraints (CausalAttribution):**
```
Answer: Q4 sales declined by 15% compared to Q3. However, I cannot determine
the causal factors from the available data. The documents mention increased
competition and seasonal patterns, but these are correlations, not confirmed causes.

Mode: QUALIFIED
```

## Dependencies

- No external dependencies
- Pure Python implementation
- Works with any LLM provider

## Related Features

- **Multi-Hop Reasoning** - Iterative retrieval can gather more evidence, reducing ABSTAIN rate
- **Hierarchical RAG** - Corpus summaries help detect when information is genuinely missing
- **Aggregation Queries** - Comprehensive retrieval reduces false negatives
