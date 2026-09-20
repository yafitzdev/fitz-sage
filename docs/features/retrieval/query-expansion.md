# Semantic Query Expansion

## Purpose

BM25 is fitz-sage's central recall mechanism, so lexical mismatch matters. A
query may use an acronym while the relevant document uses its long form, or
name one component while a diagnostic document names the component and an
associated error together.

fitz-sage builds a dynamic term graph from the indexed collection. The router
keeps the original query leg and runs graph expansions as an additional BM25
leg.

```text
original query --------------------------> BM25 leg --\
collection term-graph expansions --------> BM25 leg ----+-> fused candidate budget
other query-shape variations ------------> BM25 legs --/
```

The merged candidates still pass through the ONNX cross-encoder reranker and
Pyrrho governance decision.

## What the Index Learns

Foreground ingestion records evidence for:

- parenthetical abbreviations in both directions;
- explicit aliases and renamed terms;
- error identifiers and nearby product/component phrases;
- code and configuration identifier variants;
- repeated phrases and sentence-level co-occurrence;
- strongly connected clusters supported across the corpus.

For example, a document containing `Service Level Agreement (SLA)` makes either
form retrievable from the other. Repeated evidence connecting `MSI`, `Windows
Installer`, and `error 1722` forms a collection-specific cluster without a
global synonym table.

## Query-Time Flow

1. The deterministic planner extracts terms exactly as written by the user.
2. The SQLite index matches known forms in the query.
3. It ranks directly supported relationships and cluster neighbors.
4. fitz-sage de-duplicates expansions against the literal plan.
5. The router searches the original query and merged expansion terms as
   separate BM25 legs.

Each trace entry includes the expansion term, score, relation type, matched
source form, and supporting-document count. Expansion is bounded and
deterministic for a fixed collection state.

## Evidence Lifecycle

Semantic facts belong to their source file. Reindexing replaces that file's
facts; deletion retracts them. Aggregate relation weights and clusters are
rebuilt lazily after a corpus mutation, avoiding repeated global work during a
large ingestion run.

The graph does not claim universal equivalence. Co-occurrence is weighted below
explicit abbreviation and alias relations, and every expanded query retains
its literal retrieval leg.

## Implementation

- Extraction, aggregation, and query matching:
  `fitz_sage/engines/fitz_krag/semantic_index.py`
- Query-time merge and trace:
  `fitz_sage/engines/fitz_krag/query_pipeline.py`
- BM25 keyword leg:
  `fitz_sage/engines/fitz_krag/retrieval/router.py`

## Related

- [Sparse Search](sparse-search.md)
- [Keyword Vocabulary](keyword-vocabulary.md)
- [Managed Models](../../MANAGED_MODELS.md)
