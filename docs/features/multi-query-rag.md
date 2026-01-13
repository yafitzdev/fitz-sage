# Multi-Query RAG

## Problem

Standard RAG takes a large query (e.g., full test report + spec + requirements) and embeds it as one vector. This dilutes the semantic signal and retrieves irrelevant chunks.

## Solution: Automatic Query Expansion

Instead of embedding the entire input as one query, automatically extract key search terms and run multiple targeted queries.

## How It Works

```
Query comes in
    │
    ├─ len(query) < 300 chars? → Single vector search (standard behavior)
    │
    └─ len(query) >= 300 chars?
           │
           ▼
       Fast LLM: "Extract key search terms"
           │
           ▼
       Multiple targeted queries (3-5)
           │
           ▼
       Vector search for each → Dedupe → Rerank → Return
```

## Key Design Decisions

1. **Always-on** - No user configuration needed. Built into existing retrieval plugins.

2. **Fast LLM** - Uses `tier="fast"` model for query expansion. Cheap (~100-200ms) and negligible cost.

3. **Length-based routing** - Only triggers for queries ≥300 characters. Short queries bypass expansion entirely.

4. **LLM handles extraction** - No regex, no entity configuration. LLM figures out what's important (Jira tickets, error codes, names, etc.).

5. **Graceful degradation** - If chat client unavailable, falls back to single search.

## Example

**Input (long test report):**
```
Test TC_CAN_001 failed with error 0x4F on CAN Bus module.
The test was checking timeout behavior and got "no response" after 500ms.
Expected: ACK within 100ms. Actual: Timeout.
...
```

**LLM extracts:**
```json
["TC_CAN_001 known issues", "CAN Bus timeout", "error 0x4F", "no response timeout"]
```

**Result:** 4 targeted searches instead of 1 diluted search. Better retrieval precision.

## Configuration

No configuration required. Feature is automatically enabled in `dense` and `dense_rerank` plugins.

The threshold (300 chars) can be adjusted per-plugin in YAML:

```yaml
steps:
  - type: multi_query_search
    k: 25
    min_query_length: 300  # Adjust threshold here
```

## Benefits

| Standard RAG | Multi-Query RAG |
|--------------|-----------------|
| 1 diluted query | N targeted queries |
| Random chunks | Relevant chunks |
| LLM filters noise | LLM analyzes |
| ~30-50% precision | ~60-80% precision |

## Use Cases

- Test failure analysis (automotive, QA)
- Log analysis with structured data
- Support ticket routing with metadata
- Any scenario with long, structured input + unstructured knowledge base
