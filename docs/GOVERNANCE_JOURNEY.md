# The Governance Journey

The complete story of how Fitz learned to say "I don't know" — from hand-coded rules to a two-stage ML classifier, across 1,113 labeled cases and many experiments. This document covers the journey through the 4-class era (Chapters 1-9). For the 3-class pivot and final 90.9% result, see [fitz-gov 3.0 analysis](evaluation/fitz-gov-3.0-analysis.md).

---

## Chapter 1: The Problem

Every RAG system retrieves documents and generates answers. Most of them hallucinate confidently when the documents don't contain the answer, contradict each other, or only partially address the question.

The standard approach is to check retrieval quality: did you find the right documents? But that's the wrong question. The right question is: **given what you found, should you answer at all?**

This is epistemic governance — deciding whether to:
- **Abstain**: "I don't have the information to answer this."
- **Dispute**: "My sources contradict each other about this."
- **Qualify**: "I can partially answer, but with caveats."
- **Confident**: "Here's the answer, supported by consistent evidence."

No RAG framework measured this. We decided to.

The result is a system that scores 69.1% on calibrated epistemic governance — and when it's wrong, it over-hedges rather than over-claims. **A system that says "I'm not sure" when it should say "yes" is annoying. A system that says "yes" when it should say "I'm not sure" is dangerous.** Fitz errs on the annoying side. That's a design choice, not a limitation.

---

## Chapter 2: The Rule-Based Governor (v1.0)

The first approach was simple: run a pipeline of constraints, then use priority rules to decide the governance mode.

Five constraints were built as guardrail plugins:

| Constraint | What it detects |
|------------|----------------|
| **InsufficientEvidence** (IE) | Context doesn't match the query (wrong entity, missing aspect, low relevance) |
| **ConflictAware** (CA) | Pairwise contradictions between retrieved chunks |
| **CausalAttribution** (CAA) | Causal or speculative queries with only correlational evidence |
| **SpecificInfoType** (SIT) | Missing specific information (pricing, dates, quantities) |
| **AnswerVerification** (AV) | 3-prompt LLM jury assessing evidence sufficiency |

The governor used a priority chain:
```
IE fires "abstain"  → ABSTAIN   (highest priority)
CA fires "dispute"  → DISPUTED
Any constraint denies → QUALIFIED
Nothing fires       → CONFIDENT (lowest priority)
```

This worked for clean cases but broke on boundary cases — where multiple signals compete and the right answer depends on their interaction, not their priority.

---

## Chapter 3: Building a Benchmark (fitz-gov v1.0-v2.0)

To measure governance quality, we needed labeled test cases with known correct modes. We built **fitz-gov** — a benchmark specifically for epistemic governance.

### v1.0: 200 Hand-Crafted Cases

The first 200 cases were derived from failure analysis across early experiments. Each case provides a query, 2-4 context passages (bypassing retrieval for controlled testing), the expected governance mode, and a rationale. Split into 60 tier0 sanity cases and 141 tier1 core cases.

Initial governor accuracy on these: **72%** overall, with strong dispute detection (90%) but weak abstention (72.5%).

### v2.0: Scaling to 331 Cases

We expanded to 331 cases (+131), adding harder categories: code contexts, ambiguous queries, structured data, edge cases. This immediately exposed weaknesses — overall accuracy dropped from 72% to **63.14%**.

Then began the optimization grind.

---

## Chapter 4: 21 Experiments of Pain (fitz-gov v2.0 Era)

Over 21 experiments, we tuned every knob the governor had:

| Phase | Experiments | What we tried | Result |
|-------|-------------|---------------|--------|
| IE tuning | 001-006 | Staged pipeline, thresholds, entity extraction | 63% → 66% |
| CA scaling | 007-009 | Model analysis, SIT rate detection | 66% → 68% |
| Rule engineering | 010-011 | Dispute subordination, qualified consensus | 68% → 70% |
| Deep CA work | 012-017 | 5 approaches to better contradiction detection | All blocked by 3b model limits |
| Regex tightening | 018 | Causal attribution cleanup | 70% → 71.5% |
| IE entity fix | 019 | ALL-CAPS words, generic word filtering | Fixed relevance: 22.5% → 35% |
| NLI cross-encoder | 020 | Neural contradiction detection | **Negative result** |
| CA false positives | 021 | Numerical variance, hedging, prompt rewrites | **71.5% → 72.3%** |

**Final governor score: 72.3%** on 249 governance cases (excluding grounding/relevance). But it came with brutal tradeoffs:
- Qualification swung between 56-79% across runs due to LLM nondeterminism
- Abstention stuck at 54% (decoy data problem)
- The 3b model was the bottleneck — we'd extracted everything the architecture could give

**Key lesson**: Priority rules can't handle signal interactions. When CA fires but evidence character is hedged and vector scores are high, the right answer might be "qualified" — but the rules always say "disputed" because it has higher priority.

---

## Chapter 5: The Shift to ML

The realization: constraints are good feature extractors but bad decision-makers. What if we kept the constraints running but replaced the priority rules with a trained classifier?

```
Before: Query → Constraints → Priority Rules → Mode
After:  Query → Constraints → Feature Vector → ML Classifier → Mode
```

The constraints still run identically. They become sensors instead of judges. A gradient-boosted tree learns the decision boundaries from labeled data instead of hand-coded priorities.

Why GBT:
- 1000+ examples, ~50 features = textbook tabular classification
- No GPU, no deep learning, no neural networks
- Model size: ~5MB, inference: microseconds
- Built-in feature importance (explains itself)

---

## Chapter 6: Feature Engineering

### Three Tiers of Features

**Tier 1 — Constraint metadata (free)**: Values already computed inside constraints but not surfaced. Entity match found? Aspect mismatch? Evidence character? Jury votes? We added keys to `ConstraintResult.metadata` to expose ~30 features that were being thrown away.

**Tier 2 — Vector scores (cheap)**: `mean_vector_score`, `std_vector_score`, `score_spread` — computed from chunk embeddings already available at query time. These turned out to be critical for distinguishing disputed from qualified cases.

**Tier 3 — Detection flags (medium)**: `detection_temporal`, `detection_comparison`, `detection_aggregation` — from the DetectionOrchestrator that already runs during retrieval. Required threading DetectionSummary through the pipeline to reach the constraint layer.

**Context features (no LLM)**: 11 features computed from raw text — context lengths, pairwise TF-IDF similarity, contradiction word counts, number counts, query-context overlap. These turned out to dominate feature importance.

**Total: 58 features** across all tiers.

### The Feature Extraction Pipeline

`tools/governance/extract_features.py` loads fitz-gov cases, runs each constraint individually (bypassing the staged pipeline's short-circuit behavior so every feature is computed for every case), and outputs a CSV.

Critical lesson learned: constraint concurrency. 100 workers → massive rate limits, corrupted AV features. 10 workers → still rate limits. 3 workers → some errors. **1 worker, 0 errors, 15 minutes**. Sequential wins for LLM-dependent pipelines.

---

## Chapter 7: Training the Classifier (Experiments 1-7)

### Experiment 1: Baseline GBT

Single GBT on 914 cases, 47 constraint features only. **57.4% accuracy** vs 33.3% governor. Disputed recall: 28% (8/29) — terrible. Feature importance showed generic query features (word count, vocab overlap) dominating over governance signals. The classifier was learning surface patterns, not governance logic.

### Experiment 2: The Kitchen Sink

Four improvements at once:
1. +11 context features (lengths, similarities, word counts)
2. Class weighting (balanced)
3. Multi-model comparison (GBT, RF, ET, SVM, LR)
4. Hyperparameter search (RandomizedSearchCV, 600s budget)

**RF hit 71.0%** — best overall. But disputed recall was only 45%. The ensemble (stacking 3 models) reached 69% disputed recall. Context features dominated the top 20 — 11 of the top 20 were the new text-based ones.

### Experiment 3: Tighter CA Prompts

Tightened the ConflictAware constraint: narrower contradiction definition, longer chunk truncation (400→800 chars), wider numerical variance threshold (5%→15%).

**Disputed recall jumped to 76% (ensemble)** but overall accuracy dropped 1.6pp. Classic precision-recall tradeoff in governance signals.

### Experiment 4: The Distribution Shift Disaster

We'd been training on synthetic data where Tier 2/3 features (vector scores, detection flags) were all zeros — fitz-gov cases don't go through real retrieval. When we ran the classifier on real pipeline data with actual embeddings and detection, it scored **41%**. The classifier had learned to ignore Tier 2/3 features (all-zero = no variance) and fell apart when they had real values.

This was the "oh no" moment that led to the full pipeline eval.

### Experiment 5: Retrained on Real Features

Built `eval_pipeline.py` — computes real embeddings for every fitz-gov case, runs the DetectionOrchestrator, and extracts features with real Tier 2/3 values. Retrained on this data.

**RF: 68.9% accuracy, 83% disputed recall.** The Tier 2 vector features jumped to #4/#8/#9 importance. `mean_vector_score` alone contributed 7.3% — it helps distinguish disputed (high relevance, contradicting content) from qualified (moderate relevance, incomplete content).

The D→Q confusion dropped from 13/29 to 4/29. Real features matter.

### Data Expansion: fitz-gov v3.0 (+199 Cases)

At this point we had 914 cases with a confident class of only 154 (48% recall — awful). We generated 199 new cases targeting specific failure modes:
- 95 confident patterns (opposing_with_consensus, contradiction_resolved, different_framing)
- 60 subtle disputed cases (implicit contradiction, binary conflict, temporal conflict)
- 45 abstain edge cases

Independent blind validation: **93.5% agreement**. 9 mislabeled cases fixed. Total: **1,113 cases**.

### Experiment 6: The Big Retrain

Retrained on all 1,113 cases. GBT won this time (not RF):

| Mode | Recall |
|------|--------|
| Abstain | **85%** |
| Disputed | **67%** |
| Qualified | **66%** |
| Confident | **62%** |
| **Overall** | **69.1%** |

Confident recall jumped from 48% → 62% (+14pp) — the new cases worked. But disputed regressed from 83% → 67%. Investigation showed the real regression was RF 83% → 72% (-11pp, not -16pp as it appeared). The model type changed (RF→GBT), and the new cases were deliberately harder. GBT was chosen over RF because RF had 55% qualified recall — unacceptable.

### Experiment 7: Trying to Reach 70%

Two optimization attempts:

**7a — New text features**: Added hedging counts, assertive word counts, unique number ratios. GBT dropped from 69.1% to 66.8%. The features correlated with context length (already #1) and added noise. Reverted.

**7b — Longer hyperparameter search**: 600s budget instead of 200s. GBT found max_depth=2 (vs 6), scoring 60.1%. The longer search explored a different region of hyperparameter space and found a worse local optimum.

**69.1% is the ceiling with current features and data.**

---

## Chapter 8: What We Learned

### The numbers are honest

We could report higher numbers. Training on easier cases, cherry-picking test splits, or optimizing for overall accuracy while ignoring minority classes would all inflate the headline metric. We don't.

- 92% of test cases are rated "hard" — boundary cases, not softballs
- The test set is stratified — each class is proportionally represented
- We report per-class recall — a system that gets 90% overall by ignoring the disputed class is worse than one that gets 69% with balanced performance
- The governor baseline on the same data is 27% — the classifier is a +42pp improvement

### The system fails safe

When the classifier is wrong, it mostly over-hedges:
- Confident → Qualified (15 cases): Says "maybe" when it should say "yes" — annoying but safe
- Disputed → Qualified (8 cases): Hedges when it should flag a conflict — moderate risk
- Qualified → Confident (8 cases): Over-confident on hedged evidence — dangerous but rare

Over-hedging is annoying. Over-confidence is dangerous. The classifier errs toward the annoying side.

**Error severity map** (223 test cases):

```
                       PREDICTED
                abstain  confident  disputed  qualified
          ┌─────────────────────────────────────────────┐
 abstain  │   40         -          -          7        │
A         │   OK                               safe     │
C confident│   2         32          3         15        │
T         │   safe       OK         safe       safe     │
U disputed│   2          3         26          8        │
A         │   safe       RISKY      OK         moderate │
L qualified│  10          8          5         62        │
          │   moderate   DANGEROUS  safe       OK       │
          └─────────────────────────────────────────────┘

  OK = correct    safe = over-cautious    moderate = missed signal
  RISKY = wrong conflict    DANGEROUS = false confidence (8 of 223 = 3.6%)
```

The dangerous errors (qualified→confident) represent **3.6% of all predictions** — the system answers without caveats when it should hedge. Every other error mode is either safe (over-cautious) or moderate (missed signal). This is by design: the cost of a missed hedge is far lower than the cost of false confidence.

### What makes governance hard

| Class | Recall | Why |
|-------|--------|-----|
| Abstain (85%) | Strong trigger signals — IE fires, entities don't match, contexts are short |
| Disputed (67%) | CA fires on explicit contradictions, but subtle/implicit conflicts get missed |
| Qualified (66%) | Catch-all class at every boundary — 18 subcategories spanning fundamentally different situations |
| Confident (62%) | Must learn "absence of problems" — the hardest thing for a classifier to detect |

Abstain and disputed have positive signals (something fires). Confident requires the absence of signals. Qualified is everything in between.

### Context features are proxies

The top 3 features by importance are context length mean, total, and std. These are proxies: short contexts correlate with abstain, long contexts with qualified. In production, context length depends on the document corpus, not the governance mode. The classifier may be partially overfit to fitz-gov's context length distribution.

The fix path: make constraint signals richer (CA confidence scores instead of binary, evidence character breakdowns, source agreement features). Stronger governance-specific signals should eventually outrank length proxies.

---

## Chapter 9: Current State

**Shipped**: model_v3.joblib — GBT trained on 1,113 real-feature cases.

| What | Value |
|------|-------|
| Overall accuracy | 69.1% on 223 held-out test cases |
| vs Governor | +42.2pp (governor: 26.9%) |
| Features | 58 (constraints + vectors + detection + context) |
| Model size | ~5MB, microsecond inference |
| Benchmark | fitz-gov v3.0.0, 1,113 cases, 54 subcategories |

**Architecture**: Constraints run as feature extractors → 58-dimensional vector → GBT predicts mode. Governor runs in parallel as planned fallback for low-confidence predictions.

**Integration plan**: Replace `AnswerGovernor.decide()` with classifier prediction. When `max_proba < threshold`, fall back to governor's priority rules. The governor already runs (constraints are the classifier's inputs), so the fallback is free.

### What This Enables

**Auditable AI answers.** Every response carries a governance mode with a traceable decision path: which constraints fired, what features were extracted, what the classifier predicted, and with what confidence. When a user asks "why did the system say it doesn't know?", there's a concrete answer — not "the LLM decided."

**Regulated environments.** In healthcare, legal, and financial contexts, a system that silently hallucinates is a liability. A system that flags uncertainty, disputes, and insufficient evidence — and can prove it does so 85% of the time — is auditable. The governance mode becomes part of the compliance record.

**Trust calibration.** Users learn to trust systems that are honest about their limits. When Fitz says "confident," users can rely on it because the system demonstrably hedges when it's unsure. When it says "disputed," users know to check sources. The governance mode is a trust signal, not just a label.

---

## Chapter 10: Beyond 4-Class

The 4-class approach hit a ceiling at 69.1%. The breakthrough came from recognizing that confident vs qualified was inseparable with current features (max r=0.23), collapsing to a 3-class taxonomy (abstain/disputed/trustworthy), and decomposing into two binary classifiers.

The two-stage ML classifier achieved **90.9% overall accuracy** (93.7% abstain, 94.4% disputed, 89.0% trustworthy) with only 15 critical cases (false trustworthy).

For the full story of the 3-class pivot, two-stage architecture, failed and successful approaches, and the embedding distribution fix that produced the final +14pp jump, see:
- [fitz-gov 3.0 results](evaluation/fitz-gov-3.0-results.md) — What was achieved
- [fitz-gov 3.0 analysis](evaluation/fitz-gov-3.0-analysis.md) — How we got there

---

## Timeline

| Date | Milestone |
|------|-----------|
| Feb 6, 2026 | fitz-gov v1.0 (200 cases). Governor: 72% on easy cases. |
| Feb 6, 2026 | fitz-gov v2.0 (331 cases). Governor drops to 63%. |
| Feb 6-7, 2026 | 21 experiments. Governor optimized to 72.3% on 249 cases. |
| Feb 8, 2026 | Decision: replace governor rules with ML classifier. |
| Feb 8, 2026 | Feature extraction pipeline built (Tier 1-3, 58 features). |
| Feb 8, 2026 | Experiments 1-3: Baseline → context features → CA tuning. Best: 71% overall. |
| Feb 8, 2026 | Experiment 4: Distribution shift discovered. Classifier: 41% on real features. |
| Feb 8, 2026 | Experiment 5: Retrained on real features. RF: 68.9%, 83% disputed recall. |
| Feb 8, 2026 | fitz-gov v3.0 (+199 cases, 1113 total). Blind validated at 93.5%. |
| Feb 8, 2026 | Experiment 6: GBT 69.1%. Confident +14pp. Disputed regressed -11pp. |
| Feb 8, 2026 | Experiment 7: Two optimization attempts failed. 69.1% confirmed as ceiling. |
| Feb 8, 2026 | Shipped model_v3.joblib. Documentation complete. |
| Feb 9, 2026 | 3-class pivot (confident+qualified → trustworthy). Two-stage binary classifiers. 82.96%. |
| Feb 10, 2026 | Inter-chunk features, feature parity fix, safety-first thresholds. 76.5%. |
| Feb 11, 2026 | Embedding distribution fix. **90.9%** (93.7/94.4/89.0). 15 critical cases. |

---

## Files

| File | Purpose |
|------|---------|
| `fitz_ai/governance/constraints/` | The 5 constraints (IE, CA, CAA, SIT, AV) |
| `fitz_ai/governance/constraints/feature_extractor.py` | Runtime feature extraction (50 features) |
| `fitz_ai/governance/decider.py` | GovernanceDecider (production two-stage classifier) |
| `tools/governance/extract_features.py` | Offline feature extraction from fitz-gov cases |
| `tools/governance/train_classifier.py` | Two-stage training with hyperparameter search |
| `tools/governance/calibrate_thresholds.py` | Threshold sweep for critical case minimization |
| `tools/governance/data/model_v5_calibrated.joblib` | Production model artifact |
| `tools/governance/data/features.csv` | 1113 rows x 50 columns |
| `docs/evaluation/fitz-gov-3.0-results.md` | Benchmark results |
| `docs/evaluation/fitz-gov-3.0-analysis.md` | Technical analysis |
| `docs/evaluation/RESEARCH_NOTEPAD.md` | Detailed experiment log |
| `docs/features/governance-benchmarking.md` | Public-facing feature documentation |

---

*No other open-source RAG framework publishes governance accuracy numbers. Not because they can't — because the numbers would be embarrassing. Fitz measures it, publishes the numbers, and works to improve them. That's the difference between claiming honesty and proving it.*
