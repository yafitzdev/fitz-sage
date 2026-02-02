# Two-Pass LLM Validation Plan

## Goal

Replace brittle regex-only evaluation with a two-pass system:
1. **Pass 1 (Regex)**: Fast pattern matching catches obvious violations
2. **Pass 2 (LLM)**: Semantic validation reduces false positives from regex

## Architecture

```
Response
    │
    ▼
┌─────────────────┐
│  Regex Check    │  ← Fast, deterministic
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
  PASS      FAIL
    │         │
    ▼         ▼
   ✓ Valid  ┌─────────────────┐
            │  LLM Validator  │  ← Semantic verification
            │  (Ollama)       │
            └────────┬────────┘
                     │
                ┌────┴────┐
                │         │
              PASS      FAIL
                │         │
                ▼         ▼
         False Positive  True Violation
         (Response OK)   (Response Bad)
```

## Why Two-Pass?

| Scenario | Regex Only | Two-Pass |
|----------|------------|----------|
| "$5 million revenue" (hallucinated) | ✓ Caught | ✓ Caught |
| "no $ amount specified" | ✗ False positive | ✓ LLM clears it |
| "five million dollars" (hallucinated) | ✗ Missed | ✗ Still missed* |
| "the CEO, John Smith" (hallucinated) | ✓ Caught | ✓ Caught |
| "unlike John Smith's approach" | ✗ False positive | ✓ LLM clears it |

*Future: Add LLM pre-check for high-value categories

## Implementation

### File Structure

```
fitz_ai/evaluation/
├── benchmarks/
│   ├── fitz_gov.py           # Modify: integrate validator
│   └── llm_validator.py      # NEW: LLM validation logic
└── config/
    └── evaluation.yaml       # NEW: evaluation settings
```

### Step 1: Create LLM Validator Module

**File**: `fitz_ai/evaluation/benchmarks/llm_validator.py`

```python
@dataclass
class ValidationResult:
    is_violation: bool
    confidence: float  # 0.0-1.0
    reasoning: str
    cached: bool = False

class OllamaValidator:
    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        cache_dir: Path | None = None,
    ): ...

    def validate_forbidden_claim(
        self,
        response: str,
        matched_text: str,
        pattern: str,
        context: str,
        query: str,
        rationale: str,
    ) -> ValidationResult: ...

    def validate_missing_element(
        self,
        response: str,
        required_element: str,
        context: str,
        query: str,
    ) -> ValidationResult: ...
```

### Step 2: Prompt Templates

**Forbidden Claim Validation** (grounding):
```
You are evaluating whether a RAG system hallucinated information.

CONTEXT PROVIDED TO SYSTEM:
{context}

QUESTION ASKED:
{query}

SYSTEM'S RESPONSE:
{response}

FLAGGED TEXT: "{matched_text}"
PATTERN: {pattern}
WHY FLAGGED: {rationale}

The regex flagged "{matched_text}" as a potential hallucination.

Determine if this is:
1. TRUE_VIOLATION - The system invented/hallucinated this information (not in context)
2. FALSE_POSITIVE - The flagged text is legitimate (e.g., saying "no price mentioned", quoting the context, or discussing what's NOT known)

Respond in JSON:
{
  "verdict": "TRUE_VIOLATION" | "FALSE_POSITIVE",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

**Required Element Validation** (relevance):
```
You are evaluating whether a RAG system properly acknowledged missing information.

CONTEXT PROVIDED TO SYSTEM:
{context}

QUESTION ASKED:
{query}

SYSTEM'S RESPONSE:
{response}

EXPECTED ACKNOWLEDGMENT: The response should indicate that "{required_element}" is not available/specified in the context.

Determine if the response:
1. ACKNOWLEDGES - Properly notes the information is missing/not specified
2. FAILS_TO_ACKNOWLEDGE - Does not mention the missing information, or wrongly claims to have it

Respond in JSON:
{
  "verdict": "ACKNOWLEDGES" | "FAILS_TO_ACKNOWLEDGE",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
```

### Step 3: Caching Strategy

Cache LLM responses to avoid redundant inference:

```python
cache_key = hash(f"{response}|{matched_text}|{pattern}|{context}|{query}")
```

**Cache location**: `~/.fitz/cache/llm_validation/`

**Cache format**: JSON files with TTL (default 7 days)

**Why cache?**
- Same response patterns appear across test runs
- Benchmark re-runs shouldn't re-validate identical cases
- Debugging: inspect what LLM decided

### Step 4: Configuration

**File**: `fitz_ai/evaluation/config/evaluation.yaml`

```yaml
fitz_gov:
  validation:
    # Two-pass validation
    llm_validation: true
    llm_validator:
      provider: ollama
      model: qwen2.5:14b
      base_url: http://localhost:11434
      timeout: 30.0

    # Cache settings
    cache_enabled: true
    cache_ttl_days: 7

    # Fallback behavior
    on_llm_error: "fail_open"  # fail_open (trust regex) | fail_closed (mark violation)

    # Categories to validate (others use regex only)
    validate_categories:
      - grounding
      - relevance
```

### Step 5: Integration with FitzGovBenchmark

Modify `fitz_gov.py`:

```python
class FitzGovEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.validator = None
        if config.llm_validation:
            self.validator = OllamaValidator(
                model=config.llm_validator.model,
                base_url=config.llm_validator.base_url,
            )

    def evaluate_grounding(
        self,
        response: str,
        case: FitzGovCase,
    ) -> GroundingResult:
        # Pass 1: Regex
        regex_violations = self._check_forbidden_claims_regex(
            response,
            case.forbidden_claims,
            case.evaluation_config,
        )

        if not regex_violations:
            return GroundingResult(passed=True, violations=[])

        # Pass 2: LLM validation (if enabled)
        if self.validator:
            confirmed_violations = []
            for violation in regex_violations:
                result = self.validator.validate_forbidden_claim(
                    response=response,
                    matched_text=violation.matched_text,
                    pattern=violation.pattern,
                    context=case.contexts[0],
                    query=case.query,
                    rationale=case.rationale,
                )
                if result.is_violation:
                    confirmed_violations.append(violation)
            return GroundingResult(
                passed=len(confirmed_violations) == 0,
                violations=confirmed_violations,
            )

        # No LLM: trust regex
        return GroundingResult(passed=False, violations=regex_violations)
```

### Step 6: CLI Integration

Add flags to `fitz eval fitz-gov`:

```bash
# Use LLM validation (requires ollama running)
fitz eval fitz-gov --collection=test --llm-validation

# Specify model
fitz eval fitz-gov --collection=test --llm-validation --llm-model=qwen2.5:14b

# Disable cache (for debugging)
fitz eval fitz-gov --collection=test --llm-validation --no-cache
```

---

## Testing Strategy

### Unit Tests

1. `test_llm_validator.py`:
   - Mock ollama responses
   - Test prompt generation
   - Test caching
   - Test timeout handling

2. `test_two_pass_evaluation.py`:
   - Regex pass → no LLM call
   - Regex fail + LLM pass → false positive cleared
   - Regex fail + LLM fail → violation confirmed
   - LLM error → fallback behavior

### Integration Tests

1. **Grounding false positive test**:
   - Response: "The context does not mention a specific $ amount"
   - Regex: Flags "$"
   - LLM: Should clear as false positive

2. **Grounding true violation test**:
   - Response: "The company earned $4.2 billion"
   - Regex: Flags "$4.2 billion"
   - LLM: Should confirm violation

3. **Relevance test**:
   - Query: "What is the price?"
   - Response: "The enterprise plan includes SSO and 24/7 support."
   - Required: "not specified" or "not mentioned"
   - LLM: Should flag as failing to acknowledge

---

## Rollout Plan

### Phase 1: Core Implementation
- [ ] Create `llm_validator.py` with OllamaValidator
- [ ] Add prompt templates
- [ ] Add caching layer
- [ ] Unit tests with mocked Ollama

### Phase 2: Integration
- [ ] Modify `fitz_gov.py` evaluation logic
- [ ] Add configuration schema
- [ ] Add CLI flags
- [ ] Integration tests

### Phase 3: Validation
- [ ] Run benchmark with regex-only, record results
- [ ] Run benchmark with two-pass, compare results
- [ ] Identify cases where LLM improves accuracy
- [ ] Tune prompts based on results

### Phase 4: Documentation
- [ ] Update BENCHMARKS.md
- [ ] Add troubleshooting guide for Ollama setup
- [ ] Document cache management

---

## Dependencies

**Required**:
- `httpx` (already in project) - for Ollama API calls

**Optional**:
- `ollama` Python package - alternative to raw HTTP

**External**:
- Ollama running locally with qwen2.5:14b pulled

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Ollama not running | Graceful fallback to regex-only with warning |
| LLM timeout | Configurable timeout, fail_open default |
| LLM gives inconsistent results | Caching, temperature=0 |
| Model too slow | Cache aggressively, batch if possible |
| Model makes mistakes | Log all decisions for review, confidence threshold |

---

## Success Criteria

1. **False positive rate**: Reduce from ~15% (regex) to <5% (two-pass)
2. **Performance**: <2s average per flagged case (with caching: <100ms)
3. **Reliability**: Graceful degradation when Ollama unavailable
4. **Debuggability**: All LLM decisions logged and inspectable

---

## Out of Scope (Future)

- LLM pre-check for paraphrase detection (e.g., "five million dollars")
- Fine-tuned small model for validation
- Batch inference for speed
- Multiple LLM providers (vLLM, llama.cpp)
