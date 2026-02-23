# docs/features/retrieval/

Deep-dive documentation for fitz-ai's retrieval intelligence modules. All of these run automatically — none require configuration to enable.

| File | Feature |
|---|---|
| `hybrid-search.md` | BM25 + dense vector fusion with Reciprocal Rank Fusion |
| `temporal-queries.md` | Date-aware retrieval and freshness boosting |
| `comparison-queries.md` | Side-by-side comparison query handling |
| `multi-query-rag.md` | Parallel query expansion for broader recall |
| `multi-hop-reasoning.md` | Iterative retrieval for complex questions |
| `hyde.md` | Hypothetical Document Embeddings |
| `reranking.md` | Cross-encoder reranking (Cohere or local) |
| `query-rewriting.md` | LLM-based query reformulation |
| `query-expansion.md` | Dictionary-based synonym and acronym expansion |
| `entity-graph.md` | Entity linking and graph-based retrieval |
| `sparse-search.md` | BM25 keyword index |
| `keyword-vocabulary.md` | Domain vocabulary for keyword boosting |
| `contextual-embeddings.md` | Chunk-level contextual embedding enrichment |
| `aggregation-queries.md` | Detection and handling of aggregation queries |
| `freshness-authority.md` | Source freshness and authority scoring |

For the implementation, see `fitz_ai/retrieval/` and `fitz_ai/engines/fitz_krag/retrieval/`.
