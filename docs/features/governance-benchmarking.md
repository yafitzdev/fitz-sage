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

[fitz-gov](https://github.com/yafitzdev/fitz-gov) is a governance calibration benchmark with **1,100+ labeled test cases** across 6 categories:

| Category | What It Tests | Cases | Example |
|------|---------------|-------|---------|
| **Abstention** | Context doesn't answer the question | 237 | Query: "Tokyo population" / Context: about Osaka |
| **Dispute** | Sources make mutually exclusive claims | 196 | Source A: "Project succeeded" / Source B: "Project failed" |
| **Trustworthy Hedged** | Evidence exists but needs caveats | 360 | Causal question with only correlational data |
| **Trustworthy Direct** | Clear, consistent evidence | 254 | Direct factual answer supported by all sources |

Each test case provides a query, 2-4 context passages (bypassing retrieval for controlled testing), and the expected governance mode with a rationale.

### Case Difficulty

The benchmark is deliberately hard. 92% of cases are rated "hard" — boundary cases where multiple governance modes could plausibly apply. This is by design: easy cases (query about cats, context about dogs = abstain) aren't useful for distinguishing systems. The hard cases are where governance matters.

54 subcategories cover the full taxonomy of epistemic challenges: implicit contradictions, temporal conflicts, statistical disagreements, resolved contradictions, partial evidence, wrong entities, cross-domain evidence gaps, and more.

### Data Lineage

The cases were generated in 4 rounds with independent blind validation at each stage:

| Round | Cases | Method | Validation |
|-------|-------|--------|------------|
| 1 | 200 | Hand-crafted from 21 experiments | Expert review |
| 2 | +525 | LLM-assisted boundary sampling | 95.4% blind agreement |
| 3 | +123 | Targeted edge cases (code, adversarial) | 94% blind agreement |
| 4 | +199 | Classifier failure-mode targeting | 93.5% blind agreement |

Blind validation means an independent LLM labels each case without seeing the expected answer. Cases below 90% agreement are reviewed, relabeled, or removed.

## How Fitz Classifies Governance

Fitz uses a **two-stage approach**: epistemic constraints extract features, then a trained two-stage ML classifier makes the final decision.

### Layer 1: Constraint Pipeline (Feature Extraction)

Five constraints run on every query, each contributing diagnostic signals:

| Constraint | What It Detects | Key Features |
|------------|----------------|--------------|
| **InsufficientEvidence** | Context-query relevance gaps | Entity match, aspect match, summary overlap |
| **ConflictAware** | Pairwise contradictions between sources | Contradiction signal, evidence character, numerical variance |
| **CausalAttribution** | Causal/predictive/speculative query types | Query type, evidence presence |
| **SpecificInfoType** | Missing specific information | Info type requested, entity mismatch |
| **AnswerVerification** | LLM jury assessment (3-prompt vote) | Jury votes |

Additionally, embedding-based features (vector scores, score distributions), inter-chunk text features (pairwise overlap, assertion density, TF-IDF similarity), and query classification features (temporal, comparison, aggregation detection) feed into the classifier.

**Total: 50 features** across constraint metadata, retrieval scores, text analysis, and query classification.

### Layer 2: Two-Stage ML Classifier (Decision)

A two-stage binary classifier replaces the hand-coded priority rules that previously made governance decisions.

```
Stage 1: Extra Trees — Answerability
    "Can the evidence answer this query?"
    → NO  → ABSTAIN
    → YES → pass to Stage 2

Stage 2: Random Forest — Conflict Detection
    "Do the sources conflict?"
    → YES → DISPUTED
    → NO  → TRUSTWORTHY
```

The TRUSTWORTHY output drives two response behaviors at runtime: if constraints fired during processing, the response is hedged (trustworthy_hedged cases test this); if no constraints fired, the response is direct (trustworthy_direct cases test this).

Why two-stage over 4-class:
- **Direct vs hedged is inseparable** from features alone (max correlation r=0.23). Collapsing them into "trustworthy" and using constraint signals for the split is more reliable.
- **Binary classifiers are easier to calibrate.** Each stage has one threshold to tune, with clear safety trade-offs.
- **Staged approach improved accuracy by +9.4pp** over the best 4-class model.

The classifier is tiny (~5MB), runs in microseconds, and adds zero latency to the pipeline. It fails open — if anything goes wrong, it falls back to the rule-based governor.

## Current Results

**Production model: Two-stage classifier, safety-first thresholds.**

| Decision | Meaning | Recall |
|----------|---------|--------|
| **ABSTAIN** | Evidence doesn't answer the question | **93.7%** |
| **DISPUTED** | Sources contradict each other | **94.4%** |
| **TRUSTWORTHY** | Consistent, sufficient evidence | **89.0%** |

For comparison, the rule-based governor achieves 27% accuracy on the same test set (it over-predicts "disputed" after conflict detection tuning). A naive baseline that always predicts "trustworthy" would score 34%.

### Safety-First Threshold Tuning

The Stage 2 threshold is tuned for safety, not balanced accuracy. At s2=0.79:
- **94.4% disputed recall** — nearly all real conflicts are caught
- **89.0% trustworthy recall** — well above the 70% usability threshold
- **15 dangerous errors** in 1,100+ cases — over-confidence (trustworthy when should be disputed) is the rarest failure mode

The most common error is over-hedging: predicting "disputed" when the answer is actually "trustworthy." This is annoying but harmless. The opposite (missing a real conflict) is dangerous but rare.

### Training History

| Experiment | Dataset | Accuracy | Key Finding |
|------------|---------|----------|-------------|
| 1 (baseline) | 914, synthetic features | 57.4% | Constraint features alone insufficient |
| 2 (+context features) | 914, synthetic | 71.0% | Context features dominate importance |
| 3 (CA tuning) | 914, synthetic | 69.4% | Tighter CA prompts: +7pp disputed, -1.6pp overall |
| 4 (real features) | 914, real | 41.0% | Distribution shift — synthetic≠real features |
| 5 (retrained on real) | 914, real | 68.9% | 83% disputed recall, vector features now important |
| 6 (expanded data) | 1113, real | 69.1% | +199 cases, confident +14pp, disputed -16pp (harder cases) |
| 7 (dead code cleanup) | 1113, 29 features | 80.7% | Removed 18 dead features, no regression |
| 8 (inter-chunk features) | 1113, 51 features | **82.1%** | TF-IDF similarity, assertion density, length CV |
| 9 (feature parity fix) | 1113, 50 features | **90.9%** | Fixed embedding distribution, removed dead feature |
| **Production** | 1113, calibrated | **93.7 / 94.4 / 89.0** | Safety-first thresholds, two-stage ET+RF |

### Why These Numbers Are Honest

We could report higher numbers. Training on easier cases, using a smaller test set, or optimizing for overall accuracy at the expense of safety would all inflate the headline metric. We don't.

- **92% of test cases are hard** — boundary cases, not softballs
- **The test set is stratified** — each class is represented proportionally, not cherry-picked
- **We report per-class recall** — a system that gets 90% overall by ignoring the disputed class is worse than one that gets 81% with balanced performance
- **Safety-first calibration** — we deliberately sacrifice trustworthy recall to maximize dispute detection

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

### 4. Inter-Chunk Text Features
Deterministic features computed from chunk text without LLM calls: pairwise TF-IDF cosine similarity, assertion density, numerical variance, chunk length coefficient of variation. These features alone improved Stage 2 accuracy by +10.5pp.

### 5. LLM Jury Verification
Three independent prompts assess evidence sufficiency. Unanimous agreement that context doesn't answer the query triggers qualification — a safety net against false confidence.

## Feature Importance

The top features the classifier relies on:

| Rank | Feature | Category | What It Captures |
|------|---------|----------|-----------------|
| 1 | Context length (mean) | Context | Proxy for evidence depth |
| 2 | Context length (total) | Context | Total evidence available |
| 3 | Context length (std) | Context | Asymmetry between sources |
| 4 | Vector score (mean) | Retrieval | Retrieval confidence |
| 5 | Disputed signal | Constraint | CA constraint fired |
| 6 | Chunk length CV | Inter-chunk | Length variation (disputed chunks have higher variance) |
| 7 | Max pairwise overlap | Inter-chunk | Text similarity between sources |
| 8 | Query word count | Query | Query complexity |
| 9 | Number variance | Context | Numerical disagreement between sources |
| 10 | Pairwise similarity (mean) | Inter-chunk | TF-IDF cosine between sources |

Context and inter-chunk features dominate because they're available for every case, while constraint signals only fire on specific patterns. Improving constraint signal quality is the path to better accuracy.

## Running the Benchmark

```bash
# Feature extraction (requires LLM provider)
python -m tools.governance.eval_pipeline --chat cohere --embedding ollama --workers 1

# Train classifier on extracted features
python -m tools.governance.train_classifier --mode twostage --time-budget 200

# Output: tools/governance/data/model_v5_twostage.joblib
```

## Files

| File | Purpose |
|------|---------|
| **Test cases** | [fitz-gov](https://github.com/yafitzdev/fitz-gov) (1,100+ cases) |
| **Constraints** | `fitz_ai/core/guardrails/plugins/` (IE, CA, CAA, SIT, AV) |
| **Feature extraction** | `fitz_ai/core/guardrails/feature_extractor.py` (50 features) |
| **GovernanceDecider** | `fitz_ai/core/guardrails/governance_decider.py` (two-stage ML) |
| **Training pipeline** | `tools/governance/train_classifier.py` |
| **Evaluation pipeline** | `tools/governance/eval_pipeline.py` |
| **Model artifact** | `tools/governance/data/model_v5_calibrated.joblib` |

## Technical Specification

For the full experimental record with training history, ablation results, and what worked/failed:

**[fitz-gov 3.0 results](../evaluation/fitz-gov-3.0-results.md)** — How we got from 26.9% (rules) to 90.9% (two-stage ML)

## Why This Matters

No other open-source RAG framework publishes governance accuracy numbers. Not because they can't — because the numbers would be embarrassing. Without explicit governance logic, a RAG system's epistemic behavior is determined by the LLM's tendencies: usually over-confident, occasionally hallucinating "I don't know" for no reason, never flagging source contradictions.

Fitz measures it, publishes the numbers, and works to improve them. That's the difference between claiming honesty and proving it.
