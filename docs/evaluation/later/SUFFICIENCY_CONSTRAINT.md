# Sufficiency Constraint Analysis

**Status**: Deferred — governance classifier will learn these patterns from 851-case test set
**Date**: 2025-02-08
**Related**: Experiments 1–21 (relevance optimization), fitz-gov v2.0/v3.0

## What Is the Sufficiency Constraint?

The sufficiency constraint detects when retrieved context is **topically related but lacks the specific information** needed to answer a query.

| Constraint | What It Detects | Example |
|---|---|---|
| **InsufficientEvidence** | Context has no relevant information at all | Query about Parkinson's with cooking context |
| **Sufficiency** (missing) | Context is topically related but lacks specific answer | Query: "What is iPhone 15 pricing?" / Context: iPhone 15 *features* |

This is the subtle gap where embedding similarity is high (retrieval looks good) but the actual data point isn't there. The pipeline currently marks these cases as "confident" because everything appears relevant on the surface.

## The 32 Relevance Failure Cases

32 relevance test cases, 11 passing (35%), 21 failures from the sufficiency gap.

### Failure Patterns

**Pattern 1: Wrong info type (7 cases)**
- "What is the WARRANTY?" → Context has features, motor, speed (no warranty)
- "What is the INTEREST RATE?" → Context has loan terms, repayment (no rate %)
- "What MECHANISM reduces inflammation?" → Context has clinical results (no mechanism)
- "What is the BUDGET?" → Context has campaigns (no $)
- "What is the LOAD CAPACITY?" → Context has safety instructions (no tonnage)
- "What CERTIFICATIONS?" → Context has uptime, encryption (no SOC/ISO)
- "EXACT fuel efficiency in MPG?" → Context says "excellent fuel economy" (no number)

**Pattern 2: Scope/granularity mismatch (4 cases)**
- "Unemployment in Austin?" → Context: Texas state-level data
- "Springfield population?" → Context: Illinois state + other cities (not Springfield)
- "HOURLY rates?" → Context: project-based pricing ($50K-$500K)
- "EMEA revenue?" → Context: APAC region data

**Pattern 3: Decision/recommendation questions (3 cases)**
- "Should we proceed with acquisition?" → Context has facts, no recommendation
- "Is this worth the risk?" → Context has metrics, no assessment
- "Which candidate should we hire?" → Context has descriptions, no recommendation

**Pattern 4: Prerequisite/ambiguity (3 cases)**
- Query is underspecified, context has conditional answers

**Patterns 5–9: Other**
- Bias/causation questions, format mismatch, temporal misalignment, etc.

### Detection Coverage

- Patterns 3, 4, 5 are regex-detectable without LLM
- Pattern 1 overlaps with SIT's domain
- Patterns 2, 6, 7 overlap with InsufficientEvidence's scope

## Existing Partial Sufficiency Systems

### SpecificInfoType (SIT) Constraint

Located in the STAGE_SUFFICIENCY slot. Detects 8 hardcoded info types via regex:
- Pricing, Quantity, Temporal, Specification, Measurement, Warranty, Rate, Decision

Misses 18/21 failing cases due to detection coverage gaps.

### InsufficientEvidence Aspect Classifier

Detects medical/scientific aspects (cause, symptom, treatment, etc.) and checks if chunks have matching aspects. Limited domain coverage.

### AnswerVerificationConstraint (LLM-based, already in codebase)

Located at `core/guardrails/plugins/answer_verification.py`. Uses a 3-prompt LLM jury:
1. Direct: "Can this question be answered?"
2. Inverted: "Is context INSUFFICIENT?"
3. Completeness: "Could someone write complete answer?"

**Why it doesn't work with qwen2.5:3b:**

| Threshold | Accuracy |
|-----------|----------|
| Single call (1 NO) | 26–40% (catastrophic) |
| Majority (2+ NO) | 53.33% (terrible) |
| Unanimous (3/3 NO) | 86.67% (acceptable but almost never fires) |

The 3/3 threshold is a workaround for the 3b model's inconsistency, not a design choice.

## Why a Governance Classifier Is Better Than Hand-Coding

1. **Heterogeneous patterns**: Patterns 1–9 span different domains and detection logic — no single rule covers them
2. **Diminishing returns**: Expanding SIT regex (8 → 23+ types) increases brittleness and maintenance
3. **Context-dependent**: Whether context is "sufficient" depends on subtle query–context interactions that are inherently hard to hand-code
4. **Learns from real data**: The 851-case test set represents realistic failure modes; a classifier generalizes without manual enumeration

## Implementation Options (When Revisited)

### Option A: Fix AnswerVerificationConstraint
- Lower 3/3 threshold → breaks confidence accuracy with 3b
- Replace with cloud API call → expensive, not local-first
- **Not recommended**

### Option B: Governance Classifier (Recommended)
- Train on 851 cases: "Is retrieved context sufficient to answer this query?"
- Learns patterns instead of hand-coding
- Can use cloud API for training, lightweight local inference
- Fits local-first philosophy

### Option C: Expand Hand-Coded Rules
- 2–3 hours: Additional SIT regex (captures ~5–8 more cases)
- 6–8 hours: Hybrid query-intent extraction + presence verification
- **Not recommended** — diminishing returns vs classifier

## Test Set Context

**851-case distribution (v3.0):**

| Category | Count | Notes |
|----------|-------|-------|
| Abstention | 156 | Decoy/irrelevant data |
| Qualification | 318 | Dispute↔Qualify boundary |
| Dispute | 109 | Contradiction detection |
| Confidence | 142 | High-confidence answers |
| Grounding | 34 | Answer quality (90.5% baseline) |
| Relevance | 32 | 21 failures from sufficiency gap |

**Key additions in v3.0:**
- Dispute↔Qualify boundary: 140 cases across 14 types (primary bottleneck)
- Three-way ambiguity: 90 cases (previously 0)
- Abstain boundaries: 65 cases
- Confident boundaries: 45 cases

## Testing Strategy

- **Quick regression**: v2.0 subset (249 cases, ~2 min) — run on every commit
- **Full suite**: 851 cases — nightly or pre-release
- **Comparability**: Don't mix Ollama (3b) with cloud models in same benchmark run
