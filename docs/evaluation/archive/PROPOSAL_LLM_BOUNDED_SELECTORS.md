# Proposal: Bounded LLM Selectors for Constraint System

**Date**: February 7, 2026
**Branch**: `refactor/staged-constraint-pipeline`
**Baseline**: 70.3% fitz-gov 2.0 governance accuracy (249 cases)

---

## The Pattern

When a deterministic heuristic (regex, embedding similarity) is uncertain or produces
incomplete results, ask an LLM to choose from a **closed set** of deterministic candidates.
The LLM cannot invent answers — only select from pre-computed options or answer NONE.

This gives LLM reasoning without hallucination risk.

### Properties

1. **Closed set only** — LLM selects from deterministic candidates
2. **Deterministic validation** — reject list filters out generic/abstract selections
3. **Graceful degradation** — no chat client = silent fallback to heuristic-only
4. **Precision filter** — fires after heuristic, not instead of it

### Proven: Experiment 014 (Primary Entity Extraction)

Implemented in `insufficient_evidence.py`. When regex-based entity extraction returns an
empty primary set, LLM selects from specific entities + noun phrases. 5/6 correct, 1/6
correct NONE. Zero false selections. Zero benchmark regression.

---

## Proposal #1: SIT Entity-Relevance Verifier

**Priority**: Highest (best impact-to-risk ratio)

### Problem: The Decoy Data Pattern

Documented in Experiment 013. SIT's `_check_for_info_type()` uses regex to find answer-form
patterns (percentages, dollar amounts, rates) in chunk text. It correctly identifies the
answer form but cannot determine whether the data is for the **right entity**.

| Query | Expected | Chunk Content | SIT Behavior |
|-------|----------|---------------|--------------|
| "Capital gains tax rate?" | abstain | Income tax 10%-37%, corporate 21% | Finds `%` → ALLOW |
| "Average salary in Austin?" | abstain | National avg $120k, California $145k | Finds `$` → ALLOW |
| "Customer retention rate?" | abstain | Revenue YoY growth percentages | Finds `%` → ALLOW |

### Bounded Selector Design

**When**: After regex finds a matching info-type pattern in chunks (positive match).
**Closed set**: `{YES, NO}`
**Prompt**: "Is this [info_type] data specifically about [query_subject]?"
**Query subject**: Already extracted by `_extract_query_subject()` (existing method).

```
_check_for_info_type() finds regex match
        │
  chat available? ─No─→ return True (current behavior, graceful degradation)
        │
       Yes
        │
  Extract query subject via _extract_query_subject()
  Extract matched snippet from chunk text (~100 chars around regex match)
        │
  LLM: "Is this [rate/price/quantity] specifically about [subject]? YES or NO"
        │
  YES → return True (info found for right entity)
  NO  → return False (decoy data, trigger 'qualified')
```

### Expected Impact

Targets 23 remaining `abstain→confident` and `qualified→confident` failures identified as
"entity discrimination problems" in Exp 013. Conservatively **+5 to +8 cases**.

### Risk: Low

- Only fires as precision filter after regex already matched
- False negatives (LLM says NO wrongly) → SIT fires `qualified` instead of `allow` (safe direction)
- Cost: ~1 LLM call per case where SIT detects info type AND finds regex match

### Files to Modify

- `fitz_sage/core/guardrails/plugins/specific_info_type.py` — add `chat` param, LLM verification in `_check_for_info_type()`
- `fitz_sage/core/guardrails/__init__.py` — pass `chat` to SIT in factory
- `fitz_sage/evaluation/benchmarks/fitz_gov.py` — pass `fast_chat` to SIT
- `run_targeted_benchmark.py` — pass `fast_chat` to SIT

---

## Proposal #2: Governance Dispute Disambiguator

**Priority**: Highest absolute upside, but highest risk

### Problem: Lone CA Fires

19 `qualified→disputed` failures remain. All are lone ConflictAware fires where the
governor has no way to distinguish true contradictions from complementary perspectives.

Experiment 009 diagnostic: 16/20 false positive disputes are "Pattern A: only CA fires,
qualified_count=0, disputed_count=1" — and 4/5 correct disputes look structurally identical.

Three model-level approaches failed (Exp 007 fusion, Exp 012 scaling, Exp 012b routing)
because the 3b model's true positives and false positives are drawn from the same pool.

### Bounded Selector Design

**When**: Only when ConflictAware is the ONLY constraint that fired `disputed` (lone CA).
**Closed set**: `{GENUINE_CONTRADICTION, COMPLEMENTARY_PERSPECTIVES}`
**Prompt**: Shows the two chunk excerpts CA compared + the query.

```
AnswerGovernor.decide() detects lone CA fire
        │
  chat available? ─No─→ current behavior (disputed)
        │
       Yes
        │
  Extract chunk texts from CA metadata
  (requires CA to pass compared_texts in ConstraintResult metadata)
        │
  LLM: "Are these GENUINELY CONTRADICTORY or COMPLEMENTARY PERSPECTIVES?"
        │
  CONTRADICTION → keep disputed
  COMPLEMENTARY → downgrade to qualified
```

### Expected Impact

If LLM achieves 70% accuracy on the binary classification: ~11 of 19 fixed, ~1-2 broken.
Net **+9 to +10 cases**. Qualification category 55.9% → ~70%.

### Risk: Medium-High

- Operates on dispute signal at 87.3% recall — regression is costly
- Three prior attempts on this same problem space failed at model level
- Requires CA metadata passthrough (chunk texts not currently in ConstraintResult)
- Different approach though: classifying *type* of contradiction, not *detecting* it

### Files to Modify

- `fitz_sage/core/guardrails/plugins/conflict_aware.py` — include compared texts in result metadata
- `fitz_sage/core/governance.py` — add LLM disambiguation in `_resolve_mode()`
- Multiple factory/benchmark files for `chat` wiring

---

## Proposal #3: Aspect Classifier LLM Fallback

**Priority**: Medium (compounds with existing IE aspect check)

### Problem: GENERAL Fallthrough

`AspectClassifier.classify_query()` uses cascading regex across 11 aspect categories.
When no regex matches, it returns `QueryAspect.GENERAL`. This **completely disables**
the aspect mismatch detection in IE:

```python
# insufficient_evidence.py line 802
if should_check_aspect and query_aspect != QueryAspect.GENERAL:
    # aspect mismatch logic — SKIPPED when GENERAL
```

Natural formulations miss the regex patterns:
- "What mechanisms drive Alzheimer's progression?" → GENERAL (no match for PROCESS/CAUSE)
- "Where is CRISPR being deployed?" → GENERAL (APPLICATION pattern doesn't match)
- "How do doctors manage migraine patients?" → GENERAL (TREATMENT pattern doesn't match)

### Bounded Selector Design

**When**: Only when regex returns `GENERAL` (all patterns failed).
**Closed set**: `QueryAspect` enum values (CAUSE, EFFECT, SYMPTOM, TREATMENT, DEFINITION, PROCESS, APPLICATION, PRICING, COMPARISON, TIMELINE, PROOF, GENERAL).
**Prompt**: "What aspect does this question ask about? Pick ONE from the list."

```
classify_query() regex → GENERAL
        │
  chat available? ─No─→ return GENERAL (current behavior)
        │
       Yes
        │
  LLM: "Pick ONE aspect: CAUSE, EFFECT, SYMPTOM, TREATMENT, ..., or GENERAL"
        │
  Validate response against QueryAspect enum
  Return matched aspect or GENERAL if invalid
```

### Expected Impact

Enables aspect mismatch detection for queries currently falling through to GENERAL.
**+3 to +5 cases**, mostly in abstention category (currently 57.1%).

### Risk: Low-Medium

- Only fires when regex already failed
- Worst case: spurious aspect mismatch → `qualified` not `confident` (safe direction)
- Dangerous error (misclassify GENERAL as specific) requires ALL chunks to conflict (high bar)

### Files to Modify

- `fitz_sage/core/guardrails/aspect_classifier.py` — add `chat` param, LLM fallback
- IE already uses `_aspect_classifier` — just needs the classifier to accept chat

---

## Proposal #4: Causal Evidence Verifier

**Priority**: Low (marginal gain)

### Problem: Keyword-Only Causal Detection

`CausalAttributionConstraint._has_causal_evidence()` scans for 17 keywords ("because",
"due to", "caused by", etc.). Misses nuanced causal language:

- "The decline can be traced to habitat loss" ("traced to" not in list)
- "Deforestation drives species extinction" (implicit causation)

### Bounded Selector Design

**When**: Causal query detected but keyword scan finds NO evidence (false negative path).
**Closed set**: `{YES, NO}`
**Prompt**: "Does this text provide a causal explanation for [query]?"

### Expected Impact

**+1 to +3 cases**. CausalAttribution fires `qualified` not `abstain/disputed`, so errors
only shift between `qualified` and `confident` — lowest severity boundary.

### Risk: Low

- Only fires as fallback when keywords miss
- Marginal gain given comprehensive keyword list

### Files to Modify

- `fitz_sage/core/guardrails/plugins/causal_attribution.py` — add `chat` param, LLM fallback

---

## Impact Summary (Updated Feb 7, 2026)

| # | Proposal | Est. Cases | Actual Result | Status |
|---|----------|-----------|---------------|--------|
| 1 | SIT Entity-Relevance Verifier | +5 to +8 | **-3 (69.1%)** | Failed (Exp 015) |
| 2 | Governance Dispute Disambiguator | +9 to +10 | — | Skipped (3b limitation) |
| 3 | Aspect Classifier LLM Fallback | +3 to +5 | **-6 (67.9%)** | Failed (Exp 016) |
| 4 | Causal Evidence Verifier | +1 to +3 | — | Skipped (3b limitation) |

**Conclusion**: The bounded LLM selector pattern with qwen2.5:3b only works for
evidence-grounded selection from concrete candidates (Exp 014). It fails for both
binary verification (Exp 015) and abstract category classification (Exp 016).
Further accuracy improvements require non-LLM approaches or a larger model.

---

## Cross-Cutting Concern: Abstraction

All four proposals follow the same structural pattern:

1. Deterministic heuristic runs first (regex, keyword, embedding)
2. On uncertain/empty result, check if `chat` is available
3. Build prompt with closed candidate set from deterministic inputs
4. LLM selects from candidates or answers NONE
5. Validate response against candidate set + reject list
6. Return validated result or fall back to heuristic answer

This raises the question: should this be abstracted into a shared utility?

See "Abstraction Assessment" section below.

---

## Abstraction Assessment

### The Recurring Pattern

Every proposal (and Exp 014) follows this flow:

```
heuristic_result = regex_or_embedding(input)
if heuristic_result.uncertain and chat:
    candidates = build_candidates(input)           # deterministic
    candidates = filter_candidates(candidates)      # reject list
    prompt = format_prompt(candidates, context)     # template
    response = chat.chat([{"role": "user", "content": prompt}])
    validated = validate_response(response, candidates)  # closed-set check
    return validated or heuristic_result
return heuristic_result
```

### Differences Between Proposals

| Aspect | Exp 014 (IE Entity) | #1 (SIT Verifier) | #2 (Dispute) | #3 (Aspect) | #4 (Causal) |
|--------|--------------------|--------------------|--------------|-------------|-------------|
| Candidate type | Entity strings | YES/NO binary | Binary label | Enum values | YES/NO binary |
| Candidate source | specific_entities + noun phrases | Hardcoded | Hardcoded | QueryAspect enum | Hardcoded |
| Trigger condition | primary set empty | regex found match | lone CA fire | regex → GENERAL | keywords missed |
| Validation | match in candidates + reject set | contains YES/NO | contains keyword | match enum value | contains YES/NO |
| Context in prompt | Query + candidates | Query + subject + snippet | Query + 2 chunk texts | Query only | Query + chunk text |

### Should We Abstract?

**Arguments for abstraction:**
- DRY: the chat-call + validate + fallback logic repeats
- Consistent error handling (try/except, logging, graceful degradation)
- Consistent prompt structure
- Makes adding future bounded selectors trivial

**Arguments against abstraction:**
- Each proposal has different candidate types (strings, binary, enum)
- Each has different trigger conditions (can't generalize "when to fire")
- Each has different prompt templates (context varies significantly)
- Each has different validation logic (string match vs contains vs enum)
- Only 4-5 uses — premature abstraction for so few instances
- The actual shared code is ~10 lines (chat call + try/except + strip + validate)

### Recommendation: Don't Abstract Yet

The shared surface is too thin. What's actually common:
1. `try: response = chat.chat([{"role": "user", "content": prompt}]).strip()` — 1 line
2. `except Exception: return fallback` — 2 lines
3. Logging — 2 lines

That's ~5 lines of boilerplate per use. An abstraction would need to handle:
- Generic candidate types (`set[str]` vs `Enum` vs binary)
- Generic validation (exact match, contains, enum lookup, reject list)
- Generic prompt templates (different context requirements)
- Generic trigger conditions
- Generic fallback values

The abstraction would be more complex than the code it replaces. Each bounded selector
is ~20 lines of clear, self-contained logic. An abstraction framework would be ~50 lines
of generic machinery to save ~15 lines per use.

**Wait for 6+ uses.** If proposals #1-#4 are all implemented and a 5th/6th emerges, the
pattern will be concrete enough to extract a useful abstraction. Right now, the instances
are too varied in their specifics.

### What IS Worth Sharing

A simple **validation helper** could be useful without over-abstracting:

```python
def _llm_select_from_candidates(
    chat: Any,
    prompt: str,
    candidates: set[str],
    reject: frozenset[str] | None = None,
) -> str | None:
    """Ask LLM to select from closed candidate set. Returns match or None."""
    try:
        response = chat.chat([{"role": "user", "content": prompt}]).strip()
        response_lower = response.lower().strip().strip('"').strip("'").strip("-").strip()

        if not response_lower or response_lower == "none":
            return None

        for candidate in candidates:
            if candidate == response_lower or candidate in response_lower:
                if reject and candidate in reject:
                    return None
                return candidate

        return None  # Response not in candidates
    except Exception:
        return None  # Graceful degradation
```

This extracts the **validation logic** (the part that's actually identical) without
trying to generalize candidates, prompts, triggers, or fallback behavior. Each proposal
still owns its own prompt template, candidate generation, trigger condition, and what
to do with the result.

This helper could live in `fitz_sage/core/guardrails/llm_helpers.py` or similar.
But even this is optional — it's ~15 lines and only saves ~8 lines per use.
