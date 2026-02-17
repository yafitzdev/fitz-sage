# Reranking (Cross-Encoder Precision)

## Problem

Vector search optimizes for recall—finding candidates that might be relevant. But embedding similarity doesn't always correlate with true relevance:

- Semantically similar chunks may not answer the specific question
- The top-5 by vector distance aren't always the best 5 for the query
- Dense search retrieves broadly; users need precise answers

## Solution: Two-Stage Ranking (Baked In)

Use vector search for **recall** (find candidates), then cross-encoder reranking for **precision** (re-order by true relevance):

```
Query: "What's the battery warranty?"
            ↓
    Vector Search (recall)
    Returns 40 candidates
            ↓
    Cross-Encoder Rerank (precision)  ← Auto-enabled when rerank provider configured
    Re-scores each (query, chunk) pair
            ↓
    Top 5 truly relevant chunks
```

**Reranking is baked in** - it automatically activates when you configure a rerank provider. No plugin choice needed.

## How It Works

### Architecture

Reranking is a **conditional pipeline step** that runs after VectorSearchStep when a reranker is available:

```
VectorSearchStep (recall layer)
├─ Detection (temporal, aggregation, comparison, freshness)
├─ Query expansion, multi-query, keyword filtering
├─ Hybrid search (dense + sparse + RRF)
└─ Returns 40 candidates
         ↓
RerankStep (precision layer) [if rerank provider configured]
├─ Separates VIP chunks (score=1.0, always kept)
├─ Sends (query, chunk.content) pairs to cross-encoder
├─ Receives relevance scores (0.0-1.0)
├─ Re-orders by cross-encoder confidence
└─ Returns top 15
         ↓
LimitStep + ThresholdStep
└─ Returns final 5 chunks for generation
```

### Cross-Encoder vs Bi-Encoder

| Aspect | Bi-Encoder (Vector Search) | Cross-Encoder (Reranking) |
|--------|---------------------------|---------------------------|
| **Speed** | Fast (pre-computed embeddings) | Slow (query-time inference) |
| **Accuracy** | Good recall | Better precision |
| **Scale** | Millions of docs | Top 50-100 candidates |
| **How it works** | Embed query and docs separately | Process (query, doc) pairs together |

Cross-encoders are more accurate because they see query and document together, but too slow to run on the full corpus. That's why we use two stages.

## Key Design Decisions

1. **Baked in** - Automatically enabled when rerank provider is configured. No plugin choice needed.

2. **Provider-agnostic** - Interface via `RerankProvider` protocol. Swap providers without code changes.

3. **VIP preservation** - Artifacts and high-confidence chunks (score=1.0) bypass reranking entirely.

4. **Score metadata** - Adds `rerank_score` to chunk metadata for downstream filtering.

5. **Independent of detection** - Reranking doesn't know about temporal/aggregation/comparison detection—it just re-orders whatever VectorSearchStep returns.

6. **Smart skip** - Skips reranking if fewer than 20 candidates. Cross-encoder adds latency and cost with diminishing value on small pools.

## Configuration

### Enable Reranking

```yaml
# ~/.fitz/config/fitz_krag.yaml
fitz_krag:
  rerank: cohere                    # Provider - this alone enables reranking
  # retrieval_plugin: dense         # No need to change - reranking auto-injected
```

### Disable Reranking (default)

```yaml
fitz_krag:
  rerank: null                      # No provider = no reranking
  # Or simply omit the rerank line
```

### Provider Options

| Provider | Config | Model |
|----------|--------|-------|
| Cohere | `rerank: cohere` | `rerank-english-v3.0` (default) |
| Cohere (specific model) | `rerank: cohere/rerank-multilingual-v3.0` | Multilingual support |

### Pipeline Behavior

When reranking is enabled, the `dense` plugin automatically includes:

```yaml
# Automatic behavior (no config needed):
# - vector_search: k=40 (larger candidate pool)
# - rerank: k=15, min_chunks=20 (skip if < 20 candidates)
# - limit: k=5 (final limit from top_k config)
# - threshold: 0.6 (filter low-confidence chunks)
```

**Smart Skip**: If vector search returns fewer than 20 chunks (e.g., small collection or specific query), reranking is skipped to save latency and API costs. The cross-encoder's value is in narrowing a large pool - with only 20 candidates, vector similarity is sufficient.

## Files

- **RerankStep:** `fitz_ai/engines/fitz_krag/retrieval/steps/rerank.py`
- **RerankProvider protocol:** `fitz_ai/llm/providers/base.py`
- **Cohere implementation:** `fitz_ai/llm/providers/cohere.py`
- **Plugin definition:** `fitz_ai/engines/fitz_krag/retrieval/plugins/dense.yaml` (uses `enabled_if: reranker`)
- **Plugin loader:** `fitz_ai/engines/fitz_krag/retrieval/loader.py`

## Benefits

| Without Reranking | With Reranking |
|-------------------|----------------|
| Vector distance = relevance | Cross-encoder = true relevance |
| Top-k by embedding similarity | Top-k by query-document fit |
| Fast | Slightly slower but more accurate |
| All features still work | All features + precision layer |

## Example

**Query:** "What's the warranty period for the battery?"

**After VectorSearchStep (top 5 by vector similarity):**
1. "Battery specifications: 75 kWh capacity..." (score: 0.89)
2. "Warranty terms vary by component..." (score: 0.87)
3. "The battery uses lithium-ion cells..." (score: 0.85)
4. "Battery warranty: 8 years or 100,000 miles" (score: 0.83)
5. "Charging the battery takes 45 minutes..." (score: 0.82)

**After RerankStep (cross-encoder re-scored):**
1. "Battery warranty: 8 years or 100,000 miles" (rerank: 0.94)
2. "Warranty terms vary by component..." (rerank: 0.78)
3. "Battery specifications: 75 kWh capacity..." (rerank: 0.61)
4. "The battery uses lithium-ion cells..." (rerank: 0.45)
5. "Charging the battery takes 45 minutes..." (rerank: 0.32)

The cross-encoder correctly identifies the warranty-specific chunk as most relevant.

## Interaction with Other Features

| Feature | Relationship |
|---------|-------------|
| Hybrid Search | Runs before reranking; RRF fusion + cross-encoder = two-layer ranking |
| Query Expansion | Runs before reranking; all expanded results reranked together |
| Detection System | Inside VectorSearchStep; reranking is unaware of detected intent |
| Multi-hop | Reranking runs inside each hop independently |
| Entity Graph | Adds chunks to candidate pool; reranking can demote if not relevant |

## Dependencies

- Requires rerank provider (e.g., `rerank: cohere`)
- Works with all other retrieval features (hybrid search, detection, etc.)
- No separate plugin choice needed

## Related Features

- [**Hybrid Search**](hybrid-search.md) - Dense + sparse fusion (first ranking layer)
- [**Sparse Search**](sparse-search.md) - PostgreSQL full-text search component
- [**Multi-Hop Reasoning**](multi-hop-reasoning.md) - Reranking runs inside each hop
- [**Unified Storage**](unified-storage.md) - PostgreSQL stores vectors, reranking is query-time only
