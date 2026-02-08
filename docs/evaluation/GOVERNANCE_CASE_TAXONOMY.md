# Governance Case Taxonomy

Complete taxonomy of governance case types for fitz-gov benchmark generation.
Derived from 21 experiments of failure analysis on the dispute/qualify/abstain/confident classification task.

**Coverage target**: ~70 distinct case types across 4 pure categories and 6 boundary pairs.
Each row is a generation prompt for expanding fitz-gov.

---

## Pure Cases

Cases where classification is unambiguous.

### Abstain — context doesn't answer the query

| Case type | Example | Signal |
|-----------|---------|--------|
| Wrong entity entirely | Query: Parkinson's treatment. Context: Alzheimer's research | Different subject |
| Wrong domain | Query: Python (language). Context: Python (snake) | Homonym mismatch |
| Wrong version | Query: Python 3.12 features. Context: Python 3.10 docs | Version mismatch |
| Wrong jurisdiction | Query: EU privacy law. Context: US HIPAA | Region mismatch |
| Wrong time period | Query: 2026 pricing. Context: 2019 catalog | Temporal staleness |
| Wrong specificity | Query: Austin salary data. Context: national averages | Granularity mismatch |
| Wrong product | Query: iPhone 15 Pro Max specs. Context: iPhone 14 specs | Adjacent product |
| Missing data entirely | Query: phone number for support. Context: FAQ with no phone number | Information absence |
| Topic adjacent, no answer | Query: API rate limits. Context: API authentication docs | Same system, wrong section |
| Format impossible | Query: full text of law. Context: summary of law | Can't satisfy format need |

### Dispute — sources make mutually exclusive factual claims

| Case type | Example | Signal |
|-----------|---------|--------|
| Same metric, different values | Revenue: $5M vs $8M | Numerical conflict |
| Opposing conclusions | "X is effective" vs "X is ineffective" | Direct opposition |
| Contradictory dates | "Launched March 2024" vs "Launched June 2024" | Timeline conflict |
| Contradictory attribution | "A invented X" vs "B invented X" | Credit conflict |
| Contradictory status | "Project completed" vs "Project still in progress" | State conflict |
| Opposing recommendations | "Use framework X" vs "Never use framework X" | Prescriptive conflict |
| Statistical direction conflict | "Grew 10%" vs "Declined 5%" | Trend conflict |
| Binary fact conflict | "Feature is supported" vs "Feature is not supported" | Boolean conflict |

### Qualify — answer requires caveats or hedging

| Case type | Example | Signal |
|-----------|---------|--------|
| Different aspects discussed | Speed: excellent. Accuracy: poor | Multi-faceted |
| Mixed evidence | 3 studies support, 2 studies oppose | Inconclusive |
| Conditional applicability | Works for enterprise, fails for startups | Scoped validity |
| Hedged/uncertain claims | "May improve outcomes" "Preliminary results suggest" | Epistemic weakness |
| Temporal ambiguity | "Currently X" but doc is 2 years old | Freshness uncertain |
| Entity ambiguity | "Apple" — company or fruit? | Referent unclear |
| Scope ambiguity | "The project" with 3 projects in context | Underspecified reference |
| Deprecated but documented | React componentWillMount — works but deprecated | Valid with caveat |
| Partial correlation | Price increase correlates with churn, but not proven causal | Causal uncertainty |
| Small sample / weak methodology | "Survey of 12 people shows..." | Evidence quality concern |
| Source quality variance | Peer-reviewed study vs blog post | Authority mismatch |
| Multiple valid interpretations | "How is performance?" — speed? accuracy? sales? | Metric ambiguity |

### Confident — clear, consistent evidence

| Case type | Example | Signal |
|-----------|---------|--------|
| Direct factual answer | "What is X?" Context: "X is defined as..." | Explicit answer |
| Multiple sources converge | 3 chunks all state same fact | Corroborated |
| Clear procedural answer | "How do I X?" Context: step-by-step guide | Complete procedure |
| Unambiguous extraction | "What is the value?" Context: table with the value | Structured data match |
| Well-documented technical | "What language is React written in?" Context: React docs | Authoritative source |
| Clear causal explanation | "What caused the outage?" Context: post-mortem with root cause | Complete explanation |
| Quantitative answer available | "How many employees?" Context: "The company has 5,000 employees" | Exact match |

---

## Boundary Cases

Cases where classification is hard. These are the highest-value test cases.

### Abstain <-> Confident

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Decoy keywords | abstain | Shares vocabulary, different topic |
| Domain bleed | abstain | Closely related field (Parkinson's/Alzheimer's) |
| Adjacent product | abstain | Same product line, wrong model |
| Version near-miss | abstain | Same software, wrong version |
| Partial schema match | abstain | Table has some columns asked about, missing key one |
| High embedding similarity, wrong entity | abstain | Embeddings can't distinguish entities in same domain |

### Abstain <-> Qualify

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Related topic, missing specific info | qualify | Context IS about the topic, just lacks the specific answer |
| Partial answer | qualify | Has 1 of 3 things asked about |
| Adjacent entity, some overlap | qualify | iPhone 14 context for iPhone 15 query — some specs shared |
| Right topic, wrong info type | qualify | Asks for pricing, gets features of same product |
| Tangential but useful context | qualify | Doesn't answer directly but provides relevant background |

### Abstain <-> Dispute

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Irrelevant content with internal tension | abstain | Chunks contradict each other but neither answers the query |
| Off-topic contradiction | abstain | Real contradiction but about a different subject |

### Dispute <-> Qualify (primary bottleneck — 34 failures)

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Same claim, different values | dispute | Factual conflict on same measurement |
| Same topic, different aspects | qualify | Complementary info misread as conflict |
| Same claim, different time periods | qualify | Both true at different times |
| Same claim, different conditions/scope | qualify | Both true in different contexts |
| Same metric, methodology difference | qualify | Measurement approach differs |
| Directly opposing conclusions | dispute | Genuine disagreement on same question |
| Hedged claim vs assertive counterclaim | qualify | Asymmetric evidence strength |
| Numerical near-miss (rounding) | qualify | $5.0M vs $5.2M — methodology, not conflict |
| Pros vs cons of same thing | qualify | Balanced assessment, not contradiction |
| Risk vs benefit | qualify | Both true simultaneously |
| Evolving facts (old vs new) | qualify | Superseded, not contradicted |
| Source quality asymmetry | qualify | Study vs anecdote disagree |
| Implicit contradiction | dispute | Neither states it directly but implications conflict |
| Different framing, same underlying fact | confident | Apparent contradiction resolves with reading |

### Qualify <-> Confident

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Clear answer with minor edge case | confident | Exception exists but main answer is clear |
| Single authoritative source | confident | No corroboration but source is definitive |
| Old but likely still valid | qualify | Info is probably still correct but undated |
| Answer with implicit assumptions | qualify | Correct IF certain conditions hold |
| Near-complete evidence | confident | 95% of answer present, minor detail missing |
| Multiple sources agree with slight variation | confident | Phrasing differs but substance matches |

### Dispute <-> Confident

| Case type | Correct mode | Why it's confusing |
|-----------|--------------|--------------------|
| Apparent contradiction, different granularity | confident | "Revenue grew" vs "Q3 revenue dipped" — both true |
| Contradiction with clear winner | confident | One source is clearly outdated/wrong |
| Minor disagreement in otherwise clear answer | confident | Noise, not signal |

---

## Three-Way Ambiguity Cases

Cases where multiple signals compete and the correct classification depends on which signal dominates.
These are the hardest cases and where hand-tuned priority rules break down.

### Dispute <-> Qualify <-> Confident

| Case type | Correct mode | Competing signals | Why it's a three-way |
|-----------|--------------|-------------------|----------------------|
| Evolving facts with source quality asymmetry | qualify | Old peer-reviewed study says X, new blog post says Y | Dispute (different claims), qualify (source quality varies), confident (newer info wins) — depends on which signal you trust |
| Numerical difference with clear methodology explanation | confident | "$5.0M (audited)" vs "$5.2M (estimated)" | Dispute (different values), qualify (methodology caveat), confident (audited figure is definitive) |
| Opposing conclusions with consensus | confident | 9 studies say X, 1 study says Y | Dispute (contradiction exists), qualify (not unanimous), confident (overwhelming consensus) |
| Hedged contradiction with corroboration | qualify | "X may cause Y" vs "X does not cause Y" + 2 more sources support the negative | Dispute (opposing claims), qualify (hedging present), confident (corroborated negative) |

### Abstain <-> Qualify <-> Confident

| Case type | Correct mode | Competing signals | Why it's a three-way |
|-----------|--------------|-------------------|----------------------|
| Adjacent version with partial overlap | qualify | Query: React 19 hooks. Context: React 18 hooks (80% same API) | Abstain (wrong version), qualify (most info still valid), confident (shared APIs are identical) |
| Stale authoritative source | qualify | Query: current CEO. Context: 2023 annual report naming CEO X | Abstain (outdated), qualify (was true, may still be), confident (authoritative source) |
| Partial answer from definitive source | qualify | Query: full pricing. Context: official docs with only enterprise tier pricing | Abstain (missing tiers), qualify (partial), confident (what's there is authoritative) |
| Cross-domain transfer | qualify | Query: Python async best practices. Context: JavaScript async patterns | Abstain (wrong language), qualify (concepts transfer), confident (patterns are identical) |

### Abstain <-> Dispute <-> Qualify

| Case type | Correct mode | Competing signals | Why it's a three-way |
|-----------|--------------|-------------------|----------------------|
| Off-topic sources that contradict each other about a related entity | abstain | Query about Company A. Context: two chunks about Company B disagree on a metric Company A also has | Abstain (wrong entity), dispute (real contradiction), qualify (related entity provides context) |
| Wrong jurisdiction with conflicting local rules | abstain | Query: EU data law. Context: US vs California privacy laws disagreeing | Abstain (wrong jurisdiction), dispute (US vs CA conflict), qualify (privacy law concepts overlap) |
| Version mismatch with breaking changes documented | qualify | Query: API v3. Context: v2 migration guide documenting what changed in v3 | Abstain (wrong version), dispute (v2 vs v3 behavior differs), qualify (migration guide describes v3 changes) |

### Full Four-Way Ambiguity

| Case type | Correct mode | Competing signals | Why it's a four-way |
|-----------|--------------|-------------------|---------------------|
| Adjacent entity with contradictory sources and partial answer | qualify | Query about Drug A. Context: Drug A side effect data (hedged) + Drug B efficacy data (contradicts Drug A claims) | Abstain (Drug B is wrong entity), dispute (contradictory efficacy claims), qualify (hedged side effect data), confident (side effect data is from clinical trial) |
| Stale contradictory sources with partial coverage | qualify | Query: 2026 pricing. Context: 2024 pricing ($100) vs 2025 pricing ($120), both partial | Abstain (outdated), dispute ($100 vs $120), qualify (trend is informative), confident (price trajectory is clear) |

---

## Coverage Summary

| Category | Case types | Target examples each | Total target |
|----------|-----------|---------------------|-------------|
| Pure: Abstain | 10 | 10 | 100 |
| Pure: Dispute | 8 | 10 | 80 |
| Pure: Qualify | 12 | 10 | 120 |
| Pure: Confident | 7 | 10 | 70 |
| Boundary: Abstain <-> Confident | 6 | 10 | 60 |
| Boundary: Abstain <-> Qualify | 5 | 10 | 50 |
| Boundary: Abstain <-> Dispute | 2 | 10 | 20 |
| Boundary: Dispute <-> Qualify | 14 | 15 | 210 |
| Boundary: Qualify <-> Confident | 6 | 10 | 60 |
| Boundary: Dispute <-> Confident | 3 | 10 | 30 |
| Three-way ambiguity | 13 | 10 | 130 |
| **Total** | **86** | — | **~930** |

Current coverage: ~331 cases covering ~40 types (~8 per type).
Priority generation order:
1. **Dispute <-> Qualify** (14 types, 34 failures — highest ROI)
2. **Three-way ambiguity** (13 types, currently ~0 coverage)
3. **Abstain <-> Confident** (6 types, decoy data problem)
4. Everything else

---

## Generation Process

For each case type row:

1. Write 2-3 seed examples from existing failures or manually
2. Prompt a large LLM: "Generate 10 examples of [case type] across domains (medical, financial, legal, technical, consumer). Each needs: query, 2 context chunks, correct governance mode, and why."
3. Run each through the pipeline 5x — keep cases where accuracy < 60%
4. Manual verification of kept cases
5. Cross-validate: classifier trained on old+new should not regress on old test set
