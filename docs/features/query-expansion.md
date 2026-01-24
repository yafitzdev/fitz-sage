# Query Expansion (Synonym/Acronym Variations)

## Problem

Users often use different terminology than what appears in documents:

- "How do I fetch employee data?" - Document uses "retrieve" or "get"
- "How does the db connection work?" - Document uses "database"
- "What failures can occur?" - Document uses "errors" or "exceptions"

Dense semantic search helps with meaning, but exact synonym/acronym expansion improves recall.

## Solution: Lightweight Query Expansion

Expand queries with synonym and acronym variations before searching:

```
Original query:     "How do I fetch employee data?"
                              ↓
Expanded queries:   ["How do I fetch employee data?",
                     "How do I retrieve employee data?",
                     "How do I get employee data?"]
                              ↓
                    Search with all variations
                              ↓
                    RRF fusion of results
```

## How It Works

### At Query Time

1. Query is analyzed for known synonyms and acronyms
2. Up to 4 additional query variations are generated
3. Each variation is searched (with hybrid dense+sparse)
4. Results are merged using Reciprocal Rank Fusion (RRF)

### Expansion Rules

**Synonym Substitution:**
- `delete` ↔ `remove`, `erase`
- `create` ↔ `add`, `make`, `generate`
- `get` ↔ `retrieve`, `fetch`, `obtain`
- `update` ↔ `modify`, `change`, `edit`
- `error` ↔ `failure`, `exception`, `issue`
- And 40+ more common technical terms

**Acronym Expansion:**
- `api` → `application programming interface`
- `db` → `database`
- `auth` → `authentication`
- `config` → `configuration`
- `ml` → `machine learning`
- `rag` → `retrieval augmented generation`
- And 50+ more common acronyms

## Key Design Decisions

1. **Always-on** - Baked into VectorSearchStep. No configuration.

2. **Rule-based** - No LLM calls. Fast and predictable.

3. **Bidirectional synonyms** - Both directions work (fetch→retrieve, retrieve→fetch).

4. **Case-preserving** - Preserves first character case of replaced word.

5. **Limit expansions** - Maximum 4 additional variations to control latency.

6. **RRF fusion** - Same fusion method as hybrid search for consistent ranking.

## Files

- **Expansion detector:** `fitz_ai/retrieval/detection/detectors/expansion.py`
- **Integration:** `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py`

Note: Query expansion uses dictionary-based matching (not LLM) for fast, deterministic results. Synonyms and acronyms are defined in the `SYNONYMS` and `ACRONYMS` dicts in `expansion.py`.

## Benefits

| Without Expansion | With Expansion |
|-------------------|----------------|
| "fetch" misses "retrieve" docs | "fetch" finds "retrieve" docs |
| "db" misses "database" docs | "db" finds "database" docs |
| User must guess exact terms | Natural language works |
| Lower recall | Higher recall |

## Example

**Query:** "How does the db connection work?"

**Expanded to:**
1. "How does the db connection work?" (original)
2. "How does the database connection work?" (acronym expansion)
3. "How does the datastore connection work?" (synonym)

**Result:** Documents mentioning "database connection" are now found even though the user said "db".

## Performance

- Expansion is fast (microseconds, rule-based)
- Additional embedding calls (one per variation)
- Additional search calls (one per variation)
- RRF merge is fast (in-memory)

Typical overhead: 2-4x search time for 3-5 variations. Worth it for improved recall.

## Dependencies

- No LLM required (dictionary-based expansion)
- Fast, deterministic synonym/acronym matching
- To add new synonyms or acronyms, edit the dicts in `detection/detectors/expansion.py`
