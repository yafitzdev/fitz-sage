# Hybrid Search (Dense + Sparse)

## Problem

Dense (semantic) search excels at understanding meaning but can miss exact keyword matches:

- "X100 specs" - Dense search might retrieve Y200 docs if semantically similar
- "CCS Combo 1 charging" - Technical identifiers need exact matching
- "SAE Level 2+" - Acronyms might embed differently than their expansions

## Solution: Hybrid Search with RRF Fusion

Combine dense (semantic) and sparse (TF-IDF keyword) search using Reciprocal Rank Fusion:

```
Dense search:  [doc_A, doc_B, doc_C, ...]  (semantic similarity)
Sparse search: [doc_C, doc_D, doc_A, ...]  (keyword matching)
                        ↓
                   RRF Fusion
                        ↓
Combined:      [doc_A, doc_C, doc_B, doc_D, ...]  (best of both)
```

## How It Works

### At Ingestion

1. Chunks are indexed in the vector database (dense embeddings)
2. Chunks are also indexed in a TF-IDF sparse index (keyword vectors)
3. Both indices are built automatically - no configuration needed

### At Query Time

1. Query is embedded (dense vector)
2. Query is tokenized (sparse vector via TF-IDF)
3. Both indices are searched in parallel
4. Results are fused using Reciprocal Rank Fusion (RRF)

### RRF Formula

```
score(doc) = Σ 1/(k + rank_i(doc))
```

Where:
- `k` = constant (default 60, higher = more weight to lower ranks)
- `rank_i(doc)` = rank of doc in retrieval method i (1-indexed)

A document appearing at rank 1 in both dense and sparse gets:
- RRF score = 1/(60+1) + 1/(60+1) = 0.0328

A document at rank 1 in dense but rank 100 in sparse gets:
- RRF score = 1/(60+1) + 1/(60+100) = 0.0226

## Key Design Decisions

1. **Always-on** - Baked into VectorSearchStep. No plugin configuration.

2. **Graceful degradation** - If sparse index unavailable (missing scipy/sklearn), falls back to dense-only.

3. **TF-IDF over BM25** - Uses scikit-learn's TfidfVectorizer for simplicity and no additional dependencies.

4. **Built at ingestion** - Sparse index is built alongside dense embeddings, not at query time.

5. **Per-collection indices** - Each collection has its own sparse index.

## Configuration

No configuration required. Feature is baked into the retrieval pipeline.

Internal parameters in `VectorSearchStep`:
- `rrf_k`: RRF constant (default: 60)

TF-IDF vectorizer settings in `SparseIndex`:
- `sublinear_tf`: Use log(tf) for better weighting
- `ngram_range`: (1, 2) for unigrams and bigrams
- `max_df`: 0.95 (ignore terms in >95% of docs)

## Files

- **Sparse index module:** `fitz_ai/retrieval/sparse/`
- **Index storage:** `.fitz/sparse_index/{collection}.{pkl,npz,json}`
- **Ingestion hook:** `fitz_ai/ingestion/diff/executor.py` (`_build_sparse_index`)
- **Query integration:** `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py` (`_hybrid_search`)

## Benefits

| Dense Only | Hybrid (Dense + Sparse) |
|------------|-------------------------|
| Misses exact keywords | Catches exact matches |
| Semantic similarity only | Semantic + lexical |
| Acronyms may fail | Acronyms work well |
| Model numbers fuzzy | Model numbers exact |

## Example

**Query:** "X100 battery capacity"

**Dense search top-5:**
1. "The Model X100 is TechCorp's mid-range sedan..."
2. "Model Y200 features a 100 kWh battery..."
3. "Battery technology: lithium-ion cells..."

**Sparse search top-5:**
1. "Model X100: Battery capacity: 75 kWh"
2. "The Model X100 is TechCorp's mid-range sedan..."
3. "X100 pricing and specifications..."

**After RRF fusion:**
1. "Model X100: Battery capacity: 75 kWh" (exact keyword match boosted)
2. "The Model X100 is TechCorp's mid-range sedan..."
3. "X100 pricing and specifications..."

## Dependencies

- `scikit-learn` - TfidfVectorizer
- `scipy` - Sparse matrix operations
- `numpy` - Array operations

If these are not installed, hybrid search degrades gracefully to dense-only.
