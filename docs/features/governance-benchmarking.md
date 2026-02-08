# Governance Benchmarking (fitz-gov)

## The Problem with RAG Benchmarks

Standard RAG benchmarks (BEIR, MS MARCO, Natural Questions) measure **retrieval quality**: "Did you find the right documents?" Some newer benchmarks (RAGAS, TruLens) add **generation quality**: "Is the answer faithful to the retrieved context?"

Neither measures what actually matters for trust: **does the system know when it doesn't know?**

A RAG system that hallucinates confidently on every query can score perfectly on retrieval benchmarks. It found the right docs. It generated a fluent answer. The answer happens to be wrong because the docs contradict each other, but no benchmark caught that.

This is the gap fitz-gov fills.

## Why Epistemic Governance Is Harder Than Retrieval

Retrieval benchmarking has a clean signal: the document is relevant or it isn't. You can compute precision@k, recall@k, and nDCG against a gold standard. The problem is well-posed and the metrics are well-understood.

Epistemic governance has no clean signal. Consider:

- Two sources say different things. Is that a **dispute** (flag it) or a **resolved contradiction** (one supersedes the other, be confident)?
- Context is topically related but doesn't answer the specific question. Is that **abstain** (no answer possible) or **qualified** (partial answer with caveats)?
- Sources describe the same thing in different words. Is that **confident** (they agree) or **qualified** (they seem to disagree)?

These decisions require understanding *relationships between sources*, not just relevance to a query. A retrieval benchmark asks "is this document about the topic?" Governance asks "given these three documents that are all about the topic, should we trust the answer they produce?"

This is why the state of the art in governance accuracy is in the 60-75% range, while retrieval benchmarks routinely hit 90%+. The problem is fundamentally harder.

## fitz-gov: The Benchmark

[fitz-gov](https://github.com/yafitzdev/fitz-gov) is a governance calibration benchmark with **1,100+ labeled test cases** across 4 modes:

| Mode | What It Tests | Cases | Example |
|------|---------------|-------|---------|
| **ABSTAIN** | Context doesn't answer the question | 237 | Query: "Tokyo population" / Context: about Osaka |
| **DISPUTED** | Sources make mutually exclusive claims | 196 | Source A: "Project succeeded" / Source B: "Project failed" |
| **QUALIFIED** | Evidence exists but needs caveats | 360 | Causal question with only correlational data |
| **CONFIDENT** | Clear, consistent evidence | 254 | Direct factual answer supported by all sources |

Each test case provides a query, 2-4 context passages (bypassing retrieval for controlled testing), and the expected governance mode with a rationale.

### Case Difficulty

The benchmark is deliberately hard. 92% of cases are rated "hard" — boundary cases where multiple governance modes could plausibly apply. This is by design: easy cases (query about cats, context about dogs = abstain) aren't useful for distinguishing systems. The hard cases are where governance matters.

54 subcategories cover the full taxonomy of epistemic challenges: implicit contradictions, temporal conflicts, statistical disagreements, resolved contradictions, partial evidence, wrong entities, cross-domain evidence gaps, and more.

### Data Lineage

The cases were generated in 4 rounds with independent blind validation at each stage:

| Version | Cases | Method | Validation |
|---------|-------|--------|------------|
| v1.0 | 200 | Hand-crafted from 21 experiments | Expert review |
| v2.0 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| v3.0 | +123 | Targeted edge cases (code, adversarial) | 94% blind agreement |
| v4.0 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |

Blind validation means an independent LLM labels each case without seeing the expected answer. Cases below 90% agreement are reviewed, relabeled, or removed.

## How Fitz Classifies Governance

Fitz uses a **two-layer approach**: epistemic constraints extract features, then a trained ML classifier makes the final decision.

### Layer 1: Constraint Pipeline (Feature Extraction)

Five constraints run on every query, each contributing diagnostic signals:

| Constraint | What It Detects | Key Features |
|------------|----------------|--------------|
| **InsufficientEvidence** | Context-query relevance gaps | Entity match, aspect match, summary overlap |
| **ConflictAware** | Pairwise contradictions between sources | Contradiction signal, evidence character, numerical variance |
| **CausalAttribution** | Causal/predictive/speculative query types | Query type, evidence presence |
| **SpecificInfoType** | Missing specific information | Info type requested, entity mismatch |
| **AnswerVerification** | LLM jury assessment (3-prompt vote) | Jury votes |

Additionally, embedding-based features (vector scores, score distributions) and query classification features (temporal, comparison, aggregation detection) feed into the classifier.

**Total: 58 features** across 3 tiers — constraint metadata, retrieval scores, and query classification.

### Layer 2: ML Classifier (Decision)

A gradient-boosted tree (GBT) trained on the full fitz-gov dataset replaces the hand-coded priority rules that previously made governance decisions.

Why ML over rules:
- **Rules can't handle interactions.** A dispute signal + high relevance score + hedging language might mean "qualified" — rules pick "disputed" because it has higher priority.
- **Rules have a qualified bias.** The most common governance mode is "qualified" (hedging). A priority-based system defaults to hedging on anything ambiguous.
- **ML learns from 1,100+ labeled examples.** The decision boundary is empirical, not hand-tuned.

The classifier is tiny (~5MB), runs in microseconds, and adds zero latency to the pipeline.

## Current Results

**Classifier: 69.1% overall accuracy** on held-out test data (223 cases, stratified split).

| Mode | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| **Abstain** | 75% | **85%** | 80% | 47 |
| **Confident** | 73% | 62% | 67% | 52 |
| **Disputed** | 62% | 67% | 64% | 39 |
| **Qualified** | 67% | 66% | 66% | 85 |

For comparison, the rule-based governor achieves 27% accuracy on the same test set (it over-predicts "disputed" after conflict detection tuning). A naive baseline that always predicts "qualified" would score 38%.

### Why These Numbers Are Honest

We could report higher numbers. Training on easier cases, using a smaller test set, or optimizing for overall accuracy at the expense of minority classes would all inflate the headline metric. We don't.

- **92% of test cases are hard** — boundary cases, not softballs
- **The test set is stratified** — each class is represented proportionally, not cherry-picked
- **We report per-class recall** — a system that gets 90% overall by ignoring the disputed class is worse than one that gets 69% with balanced performance

### What the Numbers Mean in Practice

| Recall | What It Means |
|--------|--------------|
| **Abstain 85%** | When Fitz doesn't have the answer, it says so 85% of the time instead of hallucinating |
| **Disputed 67%** | When sources contradict, Fitz flags the conflict 67% of the time instead of picking a side |
| **Qualified 66%** | When evidence needs caveats, Fitz hedges 66% of the time instead of stating it as fact |
| **Confident 62%** | When the answer is clear, Fitz gives it directly 62% of the time instead of over-hedging |

The failure modes are instructive. When Fitz misclassifies:
- Disputed cases mostly get classified as qualified (hedging when it should flag a conflict — safe failure)
- Confident cases get classified as qualified (hedging when it should be direct — annoying but safe)
- Qualified cases get classified as confident (over-confidence — the most dangerous failure, but rare)

**The system fails safe.** Over-hedging is annoying but not dangerous. Over-confidence is dangerous but rare.

## Key Techniques

### 1. Pairwise Contradiction Detection
Instead of classifying each chunk independently, compare chunks pairwise:
```
"Do these two texts contradict each other about the question?"
```
This is critical because single-chunk classification can't see inter-source conflicts.

### 2. Critical Entity Matching
Years and numbered qualifiers must match exactly — high embedding similarity isn't enough:
```
Query: "2024 World Series" + Context: "2021 World Series" -> ABSTAIN
Query: "type 2 diabetes" + Context: "type 1 diabetes" -> ABSTAIN
```

### 3. Evidence Character Analysis
Classify the rhetorical stance of each source: "assertive" (states facts), "hedged" (uses qualifiers), or "mixed." Two assertive sources that disagree = dispute. One hedged source + one assertive = qualified.

### 4. Numerical Variance Detection
Numbers within 25% of each other (e.g., "23%" vs "27%") likely describe the same phenomenon with measurement noise — not a contradiction. Numbers >25% apart might be a real dispute.

### 5. LLM Jury Verification
Three independent prompts assess evidence sufficiency. Unanimous agreement that context doesn't answer the query triggers qualification — a safety net against false confidence.

## Feature Importance

The top features the classifier relies on:

| Rank | Feature | Importance | What It Captures |
|------|---------|------------|-----------------|
| 1 | Context length (mean) | 12.9% | Proxy for evidence depth |
| 2 | Context length (total) | 9.0% | Total evidence available |
| 3 | Context length (std) | 6.5% | Asymmetry between sources |
| 4 | Vector score (mean) | 6.3% | Retrieval confidence |
| 5 | Disputed signal | 5.2% | CA constraint fired |
| 6 | CA signal | 5.1% | Contradiction detection result |
| 7 | CA fired | 4.9% | Whether conflict check triggered |
| 8 | Query word count | 4.2% | Query complexity |
| 9 | Number variance | 4.2% | Numerical disagreement between sources |
| 10 | Pairwise similarity (mean) | 3.7% | How similar sources are to each other |

Context features dominate because they're available for every case, while constraint signals only fire on specific patterns. This is an area for improvement — stronger constraint signals should eventually outrank length proxies.

## Running the Benchmark

```bash
# Run governance evaluation
fitz eval fitz-gov --model ollama/qwen2.5:3b

# Run with specific collection
fitz eval fitz-gov --collection test
```

## Files

- **Test cases**: [fitz-gov](https://github.com/yafitzdev/fitz-gov) (1,100+ cases)
- **Constraints**: `fitz_ai/core/guardrails/plugins/` (IE, CA, CAA, SIT, AV)
- **Feature extraction**: `fitz_ai/core/guardrails/feature_extractor.py`
- **Training pipeline**: `tools/governance/train_classifier.py`
- **Evaluation pipeline**: `tools/governance/eval_pipeline.py`
- **Model artifact**: `tools/governance/data/model_v3.joblib`

## Technical Specification

For the full experimental record with ablation results, confusion matrices, and training history:

**[Classifier NOTEPAD](../evaluation/classifier/NOTEPAD.md)** — Living document with all 6 experiments

For future improvement plans:

**[Classifier Next Steps](../evaluation/later/CLASSIFIER_NEXT_STEPS.md)** — Roadmap to 70%+ per-class recall

## Why This Matters

No other open-source RAG framework publishes governance accuracy numbers. Not because they can't — because the numbers would be embarrassing. Without explicit governance logic, a RAG system's epistemic behavior is determined by the LLM's tendencies: usually over-confident, occasionally hallucinating "I don't know" for no reason, never flagging source contradictions.

Fitz measures it, publishes the numbers, and works to improve them. That's the difference between claiming honesty and proving it.
