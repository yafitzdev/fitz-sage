# Sparse Search (Full-Text / BM25)

## Problem

Dense (semantic) embeddings excel at meaning but fail on exact terms:

- **Q:** "Find documents mentioning X100"
- **Dense search:** Returns Y200 docs (semantically similar model numbers)
- **Expected:** Exact match on "X100"

Embeddings compress text into vectors, losing exact lexical information. Product codes, error messages, and technical identifiers need keyword matching.

## Solution: PostgreSQL Full-Text Search

Fitz uses PostgreSQL's built-in tsvector for sparse (keyword) search, combined with dense search via RRF fusion:

```
Query: "X100 battery specs"
         ↓
    ┌────┴────┐
    ↓         ↓
Dense      Sparse
Search     Search
(pgvector) (tsvector)
    ↓         ↓
[Y200,     [X100,
 Z300,      X100-Pro,
 X100]      BatterySpec]
    ↓         ↓
    └────┬────┘
         ↓
    RRF Fusion
         ↓
[X100, X100-Pro, Y200, Z300, BatterySpec]
```

## How It Works

### Zero-Maintenance Index

Unlike traditional BM25 implementations that require building and maintaining a separate index, Fitz leverages PostgreSQL's automatic tsvector generation:

```sql
-- Chunks table (simplified)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    content TEXT,
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- GIN index for fast lookup
CREATE INDEX idx_chunks_tsv ON chunks USING GIN(content_tsv);
```

**Benefits:**
- No separate index to build or sync
- Automatically updated on insert/update
- Transactionally consistent with vector data
- PostgreSQL handles stemming, stop words, ranking

### Search Modes

The SparseIndex supports three query modes:

| Mode | Function | Use Case |
|------|----------|----------|
| Natural language | `plainto_tsquery` | "battery specifications" |
| Phrase | `phraseto_tsquery` | Words must appear together |
| Websearch | `websearch_to_tsquery` | AND, OR, NOT, quotes |

**Websearch examples:**
```
"machine learning"     → Exact phrase
python AND django      → Both required
python OR ruby         → Either matches
python -java           → Exclude java
```

### RRF Fusion

Results from dense and sparse search are combined using Reciprocal Rank Fusion:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k=60` (standard constant) and `rank_i(d)` is the document's rank in search i.

This gives:
- Documents appearing in both searches get boosted
- High-ranking results in either search stay high
- No need to normalize different score scales

## Key Design Decisions

1. **Always-on** - Sparse search runs automatically alongside dense search. No configuration.

2. **PostgreSQL-native** - Uses built-in tsvector, not external BM25 libraries.

3. **Zero maintenance** - Index is auto-generated and auto-updated by PostgreSQL.

4. **RRF fusion** - Same fusion method as query expansion and multi-query.

5. **Graceful fallback** - If tsvector fails, dense-only results are returned.

## Configuration

No configuration required. Feature is baked into the hybrid search pipeline.

Internal parameters:
- `rrf_k`: RRF constant (default: 60)
- Language: 'english' (PostgreSQL text search config)

## Files

- **Sparse index:** `fitz_sage/retrieval/sparse/index.py`
- **Hybrid integration:** `fitz_sage/engines/fitz_krag/retrieval/steps/vector_search.py`
- **Schema:** `fitz_sage/vector_db/plugins/pgvector.py` (tsvector column definition)

## Benefits

| Dense Only | Dense + Sparse |
|------------|----------------|
| "X100" → Y200, Z300 | "X100" → X100, X100-Pro |
| Misses exact matches | Exact terms prioritized |
| Semantic similarity only | Lexical + semantic |
| Model codes conflated | Model codes distinguished |

## Example

**Documents:**
- Doc A: "The X100 has a 5000mAh battery with fast charging."
- Doc B: "The Y200 features similar battery capacity to other models."
- Doc C: "Battery specifications vary across the X-series lineup."

**Query:** "X100 battery"

**Dense search only:**
1. Doc B (score: 0.89) - "battery capacity" matches semantically
2. Doc C (score: 0.85) - "Battery specifications" close
3. Doc A (score: 0.82) - Exact match but embedding averaged

**Hybrid (Dense + Sparse):**
1. Doc A (RRF: 0.032) - Exact "X100" + "battery" match
2. Doc C (RRF: 0.028) - "X-series" partial + "battery"
3. Doc B (RRF: 0.025) - Semantic only, no exact terms

## Performance

- **Index overhead:** ~10-15% storage for tsvector + GIN index
- **Query latency:** <10ms for sparse search (GIN index is fast)
- **Total hybrid:** Dense search dominates latency (~50-100ms for pgvector)

## Dependencies

- PostgreSQL with tsvector support (built-in, no extension needed)
- Unified storage (pgvector plugin creates tsvector column)

## Related Features

- [**Hybrid Search**](hybrid-search.md) - Sparse search is the "sparse" half of hybrid
- [**Keyword Vocabulary**](keyword-vocabulary.md) - Complements sparse with pre-indexed identifiers
- [**Query Expansion**](query-expansion.md) - Expanded queries also run through sparse search
- [**Reranking**](reranking.md) - Cross-encoder re-scores after hybrid search fusion
