# Keyword Vocabulary (Exact Match)

## Problem

Semantic search struggles with identifiers because embeddings represent meaning, not exact strings:

- **Q:** "What happened with TC-1001?"
- **Semantic search:** Returns TC-1002, TC-1003, TC-999 (all "semantically similar")
- **User expectation:** Only TC-1001 results

Identifiers like test case IDs, ticket numbers, version strings, and function names need **exact matching**, not semantic similarity.

## Solution: Keyword Vocabulary Pre-filtering

Fitz auto-detects identifiers during ingestion and builds a vocabulary. At query time, keywords pre-filter chunks **before** semantic search:

```
Q: "What happened with TC-1001?"
     ↓
Chunks filtered to only those containing TC-1001
     ↓
Semantic search runs on filtered set
     ↓
Result: Only TC-1001 content, never TC-1002
```

## How It Works

### At Ingestion

1. **Pattern detection** - Regex patterns identify potential keywords:
   - Test cases: `TC-\d+`, `testcase_\d+`, `TEST_\w+`
   - Tickets: `JIRA-\d+`, `BUG-\d+`, `ISSUE-\d+`
   - Versions: `v?\d+\.\d+\.\d+`, `\d+\.\d+-beta`
   - Code identifiers: `[A-Z][a-zA-Z]+Service`, `\w+Controller`

2. **Vocabulary building** - Detected keywords are stored in PostgreSQL:
   ```sql
   -- keywords table (per collection database)
   CREATE TABLE keywords (
       id TEXT PRIMARY KEY,
       category TEXT NOT NULL,
       match TEXT[] NOT NULL,     -- keyword variations
       occurrences INTEGER DEFAULT 1,
       first_seen TIMESTAMPTZ DEFAULT NOW()
   );
   ```

3. **Inverted index** - Fast lookup via PostgreSQL queries

### At Query Time

1. **Query analysis** - Detect keywords in the user's question
2. **Pre-filtering** - If keywords found, filter chunk candidates to only those in vocabulary
3. **Semantic search** - Run embedding search on filtered set
4. **Variation matching** - Handle format differences automatically:
   ```
   TC-1001 → tc-1001, TC_1001, tc 1001
   JIRA-123 → jira-123, JIRA123, jira 123
   ```

## Key Design Decisions

1. **Always-on** - Baked into ingestion and retrieval. No configuration needed.

2. **Pre-filter, not post-filter** - Filtering happens before semantic search for efficiency.

3. **Soft filtering** - If no keyword matches found, fall back to pure semantic search (graceful degradation).

4. **Case-insensitive** - `TC-1001` matches `tc-1001` automatically.

5. **Delimiter-agnostic** - `TC-1001`, `TC_1001`, and `TC 1001` all match.

## Configuration

No configuration required. Feature is baked into the ingestion and retrieval pipelines.

Internal settings in `KeywordExtractor`:
- Pattern regexes for identifier detection
- Minimum keyword length (default: 3)
- Maximum keywords per chunk (default: 50)

## Files

- **Keyword vocabulary:** `fitz_ai/retrieval/vocabulary/`
- **Vocabulary storage:** PostgreSQL `keywords` table (per-collection database)
- **Query filtering:** `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py` (`_filter_by_keywords`)
- **Ingestion hook:** `fitz_ai/ingestion/diff/executor.py` (`_build_keyword_vocabulary`)

## Benefits

| Semantic Only | With Keyword Vocabulary |
|---------------|------------------------|
| TC-1001 → returns TC-1002, TC-1003 | TC-1001 → returns only TC-1001 |
| "version 2.0.1" fuzzy | "version 2.0.1" exact |
| AuthService might miss | AuthService guaranteed |
| No identifier awareness | Identifier-aware filtering |

## Example

**Query:** "What tests failed in TC-1001?"

**Semantic search (no vocabulary):**
1. "TC-1002 test results: PASS"
2. "TC-1003 failure log: timeout error"
3. "TC-1001 status: FAIL (assertion error)"

**With keyword vocabulary:**
1. "TC-1001 status: FAIL (assertion error)"
2. "TC-1001 detailed logs: line 42 failed"
3. "TC-1001 reproduction steps"

Only chunks containing the exact identifier `TC-1001` are considered.

## Detected Identifier Types

| Type | Examples |
|------|----------|
| **Test cases** | TC-1001, testcase_42, TEST_AUTH |
| **Tickets** | JIRA-4521, BUG-789, ISSUE-123 |
| **Versions** | v2.0.1, 1.0.0-beta, 3.5 |
| **Code classes** | AuthService, UserController, PaymentHandler |
| **Code functions** | handle_login(), process_payment(), validate() |
| **Model numbers** | X100, Model-Y200, SKU-4567 |

## Dependencies

- PostgreSQL + pgvector (unified storage)
- Vocabulary stored in `keywords` table per collection
- Automatically deleted when collection is dropped

## Related Features

- **Hybrid Search** - Combines keyword vocabulary with sparse (BM25) search for comprehensive exact matching
- **Query Expansion** - Handles synonyms, while keyword vocabulary handles exact identifiers
- **Multi-Query** - Long queries decomposed may contain multiple keywords to filter on
