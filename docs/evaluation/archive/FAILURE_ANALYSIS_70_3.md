# Failure Analysis at 70.3% Baseline

**Date**: February 7, 2026
**Branch**: `refactor/staged-constraint-pipeline`
**Baseline**: 70.3% (175/249 governance cases)
**Total failures**: 74

---

## Constraint Damage Summary

| Constraint | False Fires | Correct Fires | Net Damage |
|-----------|-------------|---------------|------------|
| conflict_aware | 34 | 3 | -31 |
| insufficient_evidence | 8 | 0 | -8 |
| causal_attribution | 6 | 3 | -3 |
| specific_info_type | 5 | 2 | -3 |
| (nothing fired) | — | — | 25 cases where something should have |

`conflict_aware` alone accounts for **46% of all failures**.

---

## Three Dominant Failure Clusters

### Cluster A: conflict_aware Over-Triggering (34 false fires)

**The single biggest problem.** CA detects "contradictory information" in contexts that are
actually *nuanced/uncertain* — not genuinely contradictory.

#### Subclusters

**A1: qualified→disputed (19 cases)** — Largest subcluster

Contexts have pros/cons, caveats, or uncertainty. CA calls this "contradictory" but the
correct governance is "qualified" (answer with caveats).

| Case ID | Query | Pattern |
|---------|-------|---------|
| t0_qualify_easy_006 | Why do customers leave negative reviews? | Multiple reasons ≠ contradiction |
| t0_qualify_easy_008 | Why did website traffic decrease? | Multiple causes ≠ contradiction |
| t1_qualify_medium_006 | What do users think about the new design? | Mixed opinions ≠ contradiction |
| t1_qualify_hard_015 | Is our security adequate against sophisticated attackers? | Pros/cons ≠ contradiction |
| t1_qualify_hard_016 | Does intermittent fasting improve cognitive function? | Mixed evidence ≠ contradiction |
| t1_qualify_hard_017 | Is this new cancer treatment effective? | Partial evidence ≠ contradiction |
| t1_qualify_hard_018 | Will autonomous vehicles be mainstream by 2030? | Uncertainty ≠ contradiction |
| t1_qualify_hard_019 | What causes migraines? | Multiple factors ≠ contradiction |
| t1_qualify_hard_026 | How do I use componentWillMount in React? | Deprecated info ≠ contradiction |
| t1_qualify_hard_027 | How do I make HTTP requests using request in Node.js? | Deprecated info ≠ contradiction |
| t1_qualify_hard_029 | When was the Mercury program completed? | Ambiguous date ≠ contradiction |
| t1_qualify_hard_033 | Who is the team lead? | Changing info ≠ contradiction |
| t1_qualify_hard_034 | What is the current price? | Volatile info ≠ contradiction |
| t1_qualify_hard_035 | What is the latest version? | Changing info ≠ contradiction |
| t1_qualify_hard_037 | How efficient is the process? | Multiple metrics ≠ contradiction |
| t1_qualify_hard_040 | Does having a mentor accelerate career advancement? | Mixed evidence ≠ contradiction |
| t1_qualify_hard_041 | Does exercise improve mental health? | Nuanced answer ≠ contradiction |
| t1_qualify_hard_042 | Did the price increase cause customers to churn? | Partial correlation ≠ contradiction |
| t1_qualify_hard_043 | Is hormone replacement therapy safe for menopausal women? | Risk/benefit ≠ contradiction |

**Root cause**: CA can't distinguish "genuinely contradictory facts" (A says X, B says NOT X)
from "nuanced/multifaceted information" (A says X has pros, B says X has cons).

**A2: abstain→disputed (8 cases)** — IE misses, CA fires wrong signal

These should abstain but IE doesn't detect the irrelevance, and CA fires disputed instead.

| Case ID | Query |
|---------|-------|
| t1_abstain_medium_013 | What is the current price of Bitcoin? |
| t1_abstain_hard_005 | What is the full text of the First Amendment? |
| t1_abstain_hard_017 | How do I configure authentication in React Router v6? |
| t1_abstain_hard_019 | Pricing, features, AND support for Enterprise plan? |
| t1_abstain_hard_020 | Ingredients, nutritional info, AND allergens for product? |
| t1_abstain_hard_023 | Current interest rate for 30-year mortgages? |
| t1_abstain_hard_024 | Who is the current CEO of Twitter? |
| t1_abstain_hard_036 | What is the company's return policy? |

**A3: confident→disputed (5 cases)** — CA fires on clearly answerable questions

| Case ID | Query |
|---------|-------|
| t1_confident_medium_004 | What caused the 2024 outage? |
| t1_confident_hard_001 | What is the company's competitive advantage? |
| t1_confident_hard_011 | What is the market size for the AI chip industry? |
| t1_confident_hard_019 | Average software engineer salary in San Francisco? |
| t1_confident_hard_028 | Global average temperature increase since pre-industrial? |

### Cluster B: abstain→confident (16 cases, "Decoy Data")

No constraint fires at all. Context looks topically related but doesn't answer the actual
question (wrong entity, wrong specificity, adjacent topic).

| Case ID | Query | Likely Decoy Pattern |
|---------|-------|---------------------|
| t0_abstain_easy_009 | How do I fix a leaky faucet? | Different topic entirely |
| t1_abstain_medium_009 | Treatment for Parkinson's disease? | Adjacent medical content |
| t1_abstain_medium_011 | How to use Python for web scraping? | Different Python topic |
| t1_abstain_medium_012 | Capital gains tax rate? | Income tax data (Exp 013) |
| t1_abstain_hard_006 | Exact API endpoint for user auth? | Generic API docs |
| t1_abstain_hard_007 | Chemical formula for table salt? | Different chemistry |
| t1_abstain_hard_013 | How to apply for Apple credit card? | Different Apple product |
| t1_abstain_hard_014 | Phone number for customer support? | No phone number in context |
| t1_abstain_hard_015 | Specs for iPhone 15 Pro Max? | Different phone model |
| t1_abstain_hard_018 | New features in Python 3.12? | Different Python version |
| t1_abstain_hard_022 | Average salary for SW engineers in Austin? | National/other city data |
| t1_abstain_hard_026 | Syntax for async/await in Python 3.11? | Different version |
| t1_abstain_hard_028 | Customer retention rate for each region? | Aggregate rate only |
| t1_abstain_hard_032 | Current market cap of Nvidia? | Different company data |
| t1_abstain_hard_034 | Safe to take supplement with BP medication? | Generic supplement info |
| t1_abstain_hard_035 | Can I deduct business expenses on tax return? | Different tax topic |

**Status**: SIT verifier (Exp 015) tried LLM approach — 3b model unreliable. Needs either
better heuristics or larger model. These are the hardest cases in the benchmark.

### Cluster C: causal_attribution False Fires (6 cases)

CA fires "qualified" on cases that should be confident.

| Case ID | Query | Why it's not causal |
|---------|-------|-------------------|
| t1_confident_hard_003 | How does the recommendation algorithm work? | Process, not cause |
| t1_confident_hard_004 | Should we use Kubernetes or Docker Swarm? | Comparison, not cause |
| t1_confident_hard_043 | What temp to cook chicken for food safety? | Factual, not cause |
| t1_dispute_hard_005 | What caused the Roman Empire to fall? | Historical, but should be disputed |
| t1_dispute_hard_018 | Which marketing channel has better ROI? | Comparison, not cause |
| t1_qualify_hard_042 | Did price increase cause customers to churn? | Correctly causal, but CA over-qualifies |

---

## Actionability Assessment

| Cluster | Failures | Fix Approach | Estimated Gain | Risk |
|---------|----------|-------------|----------------|------|
| **A: conflict_aware** | **34** | Tune nuance vs contradiction discrimination | **+10 to +17** | Med |
| B: decoy data | 16 | Needs >3b model or novel heuristics | +0 (blocked) | — |
| **C: causal_attribution** | **6** | Tighten keyword matching | **+3 to +5** | Low |
| Other (scattered) | 18 | Various | +2 to +5 | Varies |

### Priority: Cluster A (conflict_aware)

**Why**: 34 failures, single root cause (nuance ≠ contradiction), clear fix direction.

**Possible approaches**:
1. Teach CA to distinguish "contradictory facts" from "nuanced/multifaceted information"
2. Add a severity threshold — only fire disputed for strong contradictions
3. Give qualified priority over disputed when both fire (architectural change)
4. Reduce CA sensitivity to disagreement signals

### Secondary: Cluster C (causal_attribution)

**Why**: 6 failures, clear false positive patterns (process/comparison misclassified as causal).

**Possible approaches**:
1. Exclude process queries from causal detection
2. Exclude comparison queries from causal detection
3. Require stronger causal signal before firing

---

## Deep-Dive: conflict_aware Architecture (Feb 7, 2026)

### How CA Works

```
apply()
  → limit to top 5 chunks
  → relevance gate (embedding similarity ≥ 0.45)
  → for each pair (chunk[0] vs chunk[i]):
      → evidence character classification (hedged/assertive/mixed)
      → hedged vs hedged → skip pair
      → assertive vs assertive → base method
      → any hedged/mixed → force fusion
      → base method selected by adaptive:
          uncertainty query → fusion (3 prompts, 2/3 majority)
          factual query → standard (1 pairwise prompt)
```

### Current Config: `use_fusion=True, adaptive=True`

Adaptive routes uncertainty queries (matching `UNCERTAINTY_QUERY_PATTERNS`) to fusion,
everything else to standard pairwise.

### Key Metrics

- **CA true positives**: 48/55 dispute cases detected correctly (87% recall)
- **CA false positives**: 34 non-dispute cases incorrectly fired (precision: 48/(48+34) = 58.5%)
- In **27/34 false fires**, CA is the ONLY constraint that fires — no other signal to counter

### Evidence Character Gating: Dead Feature

Every single benchmark pair classifies as `assertive-assertive`. The hedge/mixed patterns
are too specific for real benchmark data. The hedged-hedged skip and hedge-upgrade-to-fusion
paths never fire.

### Approaches Tried and Results

**1. Fusion-only (all queries, no adaptive): 63.9% (-6.4%)**

Removed adaptive routing, forced all queries through fusion. Killed dispute recall:
- Dispute accuracy: 56.4% (31/55) vs 87.3% (48/55) baseline
- 18 disputed→confident transitions (fusion too conservative)
- Fusion requires 2/3 majority — too many true disputes fail this bar

**2. Standard + Fusion Confirmation Gate: 63.5% (-6.8%)**

After standard finds contradiction, confirm with fusion before firing. Effectively
requires 3/4 agreement (1 standard + 2/3 fusion). Even worse than fusion-only:
- Dispute accuracy: 54.5% (30/55)
- Same problem: fusion confirmation rejects true disputes

**3. Stance Pre-Filter: Not viable**

Classify each chunk's stance (YES/NO/UNCLEAR) on the query before pairwise check.
Only proceed with pairwise when stances disagree.

Small sample (10 cases): perfect discrimination — 5/5 true disputes show YES vs NO,
5/5 false fires show same-direction.

Full benchmark (55 disputes): only 49% recall. Non-polar disputes (How many? When?
What causes?) have both chunks answering with the same stance (YES/YES or UNCLEAR/UNCLEAR)
even though the answers contradict. Stance check only works for polar yes/no questions.

**4. Widen Uncertainty Patterns: Not viable**

Broader patterns catch false fires but also route 22/55 true disputes to fusion, killing
their recall. The patterns that catch qualified→disputed false fires ("does ", "when ",
"did ") also catch genuine disputes.

**5. Governor-Level Changes: Not viable**

In 27/34 false fires, CA is the only signal. Same for true disputes — CA fires alone.
Governor cannot distinguish lone-CA-true-dispute from lone-CA-false-fire.

### The Fundamental Tension

**Any approach that makes CA harder to trigger proportionally affects both true disputes
and false fires.** The 3b model cannot distinguish:

- "Revenue was $5M" vs "Revenue was $8M" → TRUE contradiction (opposing facts)
- "Studies show benefits" vs "Some limitations noted" → FALSE contradiction (perspectives)

Both look like "contradictory information" to the model. The discrimination capacity
required to tell them apart exceeds qwen2.5:3b's capability.

This is the same fundamental 3b limitation from Exp 007, 012, 015, 016 — manifested
differently (pairwise contradiction detection vs binary classification vs multi-class).

### Remaining Viable Approaches

1. **Larger model** — 7b or 14b model for CA only (the rest stays 3b)
2. **Cluster C fix** — causal_attribution false fires are fixable with regex (6 cases) **→ DONE (Exp 018): +3 cases, 70.3% → 71.5%**
3. **Accept 71.5%** as the 3b ceiling for dispute-related governance

---

## Theoretical Ceiling (Revised)

**With qwen2.5:3b only**:
- Cluster A: **Blocked** — fundamental 3b discrimination limit
- Cluster B (decoy data): **Blocked** — needs entity-relevance discrimination
- Cluster C (causal_attribution): **Fixed (Exp 018)** — +3 cases via regex tightening
- Other scattered: Partially fixable (+2 to +3)

**Current: 71.5% (178/249)** after Exp 018
**Revised realistic ceiling with 3b: ~73-75%** (from 71.5%)

**With model upgrade for CA**: 75-80%+ (removes the 3b discrimination bottleneck)
