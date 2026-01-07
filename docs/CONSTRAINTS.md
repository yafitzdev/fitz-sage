# Epistemic Constraints

How Fitz knows when to say "I don't know."

---

## Overview

Most RAG systems confidently answer even when they shouldn't. Fitz uses **epistemic constraints** to detect problematic situations and respond appropriately.

```
┌─────────────────────────────────────────────────────────────────┐
│  User Query                                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Retrieve Chunks                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Apply Constraints                                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────┐  │
│  │ Insufficient  │ │ Conflict      │ │ Causal Attribution    │  │
│  │ Evidence      │ │ Aware         │ │                       │  │
│  └───────────────┘ └───────────────┘ └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Generate Answer with Appropriate Mode                          │
│  CONFIDENT | QUALIFIED | DISPUTED | ABSTAIN                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Answer Modes

Constraints determine the answer mode:

| Mode | Signal | User sees |
|------|--------|-----------|
| **CONFIDENT** | No constraints triggered | Direct answer |
| **QUALIFIED** | Some limitations | Answer with caveats |
| **DISPUTED** | Sources conflict | "Sources disagree..." |
| **ABSTAIN** | Insufficient evidence | "I cannot find..." |

---

## Built-in Constraints

### 1. Insufficient Evidence

**Plugin:** `insufficient_evidence`

Prevents confident answers when there's not enough direct evidence.

**What it checks:**
- Are there any chunks retrieved?
- For causal questions: is there causal language in sources?
- For fact questions: are there direct assertions?

**Example:**

```
Q: "What was our Q4 revenue?"
Chunks: [Q1 revenue: $10M, Q2 revenue: $12M, Q3 revenue: $11M]

Constraint: Triggered - no Q4 data found
Mode: ABSTAIN
Answer: "I cannot find Q4 revenue figures in the provided documents.
         The available data covers Q1-Q3 only."
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

### 2. Conflict Aware

**Plugin:** `conflict_aware`

Detects when sources contradict each other.

**What it detects:**
- Mutually exclusive claims (A vs not-A)
- Contradicting values (increased vs decreased)
- Timeline conflicts (before vs after)

**Example:**

```
Q: "Was the 2023 incident a security breach?"
Chunk 1: "The 2023 incident was classified as a security breach..."
Chunk 2: "Investigation concluded this was an operational issue, not security..."

Constraint: Triggered - conflicting classifications
Mode: DISPUTED
Answer: "Sources disagree on this classification.
         - [Source 1] describes it as a security breach
         - [Source 2] classifies it as an operational issue"
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

### 3. Causal Attribution

**Plugin:** `causal_attribution`

Prevents inventing causal explanations not stated in sources.

**What it enforces:**
- Causal queries ("why", "what caused") need explicit causal language
- Won't synthesize causality from correlation
- Won't invent reasons not in the documents

**Example:**

```
Q: "Why did the deployment fail?"
Chunks: [Deployment failed at 3pm. Logs show timeout errors. System was restored at 5pm.]

Constraint: Triggered - no explicit "because" or causal language
Mode: QUALIFIED
Answer: "I can describe what happened but cannot determine the cause from
         the available documents. The deployment failed at 3pm with timeout
         errors. The documents don't explain why this occurred."
```

**Configuration:**

```yaml
# Enabled by default, no config needed
```

---

## How Constraints Work

### Constraint Result

Each constraint returns a result:

```python
ConstraintResult(
    allow_decisive_answer: bool,   # Can we give a confident answer?
    reason: str,                   # Human-readable explanation
    signal: str,                   # "abstain", "disputed", "qualified"
    metadata: dict                 # Debug info
)
```

### Execution flow

1. Retrieve relevant chunks
2. Run all constraints on (query, chunks)
3. Collect signals from any triggered constraints
4. Resolve to answer mode
5. Generate answer with appropriate framing

### Signal resolution

| Signals | Resolved Mode |
|---------|--------------|
| No constraints triggered | CONFIDENT |
| `qualified` only | QUALIFIED |
| `disputed` (any) | DISPUTED |
| `abstain` (any) | ABSTAIN |

---

## Semantic Matching

Constraints use **embedding-based semantic matching** for language-agnostic detection.

Instead of keyword matching:
```python
# Bad: Only works in English
if "because" in text or "caused by" in text:
    has_causal_language = True
```

Fitz uses embeddings:
```python
# Good: Works in any language
causal_similarity = cosine_similarity(
    embed(chunk),
    embed("This was caused by the following reason")
)
has_causal_language = causal_similarity > threshold
```

This means constraints work across:
- Multiple languages
- Paraphrased expressions
- Domain-specific terminology

---

## Creating Custom Constraints

### Protocol

Constraints implement the `ConstraintPlugin` protocol:

```python
from fitz_ai.core.guardrails.base import ConstraintPlugin, ConstraintResult

class MyConstraint:
    @property
    def name(self) -> str:
        return "my_constraint"

    def apply(self, query: str, chunks: list) -> ConstraintResult:
        # Your logic here
        if some_problem_detected:
            return ConstraintResult.deny(
                reason="Explanation for user",
                signal="qualified"  # or "disputed", "abstain"
            )
        return ConstraintResult.allow()
```

### Requirements

Constraints must be:
- **Deterministic**: Same input → same output
- **Side-effect free**: No modifications, no network calls
- **Fast**: No LLM calls (use pre-computed embeddings)

### Example: Recency Constraint

```python
from datetime import datetime, timedelta
from fitz_ai.core.guardrails.base import ConstraintResult

class RecencyConstraint:
    """Warn when sources are outdated."""

    def __init__(self, max_age_days: int = 365):
        self.max_age = timedelta(days=max_age_days)

    @property
    def name(self) -> str:
        return "recency"

    def apply(self, query: str, chunks: list) -> ConstraintResult:
        now = datetime.utcnow()
        outdated = []

        for chunk in chunks:
            created_at = chunk.metadata.get("created_at")
            if created_at:
                age = now - datetime.fromisoformat(created_at)
                if age > self.max_age:
                    outdated.append(chunk.id)

        if len(outdated) == len(chunks):
            return ConstraintResult.deny(
                reason=f"All sources are over {self.max_age.days} days old",
                signal="qualified",
                outdated_chunks=outdated
            )

        return ConstraintResult.allow()
```

### Registration

Place custom constraints in:
```
fitz_ai/core/guardrails/plugins/my_constraint.py
```

They are auto-discovered.

---

## Comparison with Other Systems

| System | Uncertainty Handling |
|--------|---------------------|
| ChatGPT RAG | No explicit handling |
| LangChain | Prompt-based, not enforced |
| LlamaIndex | Optional, manual setup |
| **Fitz** | Built-in, automatic |

Fitz treats uncertainty as a **feature**, not a failure.

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_ai/core/guardrails/base.py` | Protocol and result types |
| `fitz_ai/core/guardrails/semantic.py` | Semantic matching utilities |
| `fitz_ai/core/guardrails/plugins/` | Built-in constraints |

---

## See Also

- [ENGINES.md](ENGINES.md) - RAG engine architecture
- [CONFIG.md](CONFIG.md) - Configuration reference
- [PLUGINS.md](PLUGINS.md) - Plugin development
