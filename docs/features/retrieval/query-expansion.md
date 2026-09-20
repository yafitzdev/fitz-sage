# Semantic Query Expansion

## Purpose

BM25 is fitz-sage's central recall mechanism, so lexical mismatch matters. A
query may use an acronym while the relevant document uses its long form, use a
spaced name for a code identifier, or name one component while a diagnostic
document records that component with an error.

fitz-sage builds a deterministic term graph from each indexed collection. The
graph is stored in the collection's SQLite database and requires no language
model, global synonym dictionary, or external service. The router keeps the
original query leg and searches graph expansions as an additional BM25 leg.

```text
original query --------------------------> BM25 leg --\
collection term-graph expansions --------> BM25 leg ----+-> fused candidate budget
other query-shape variations ------------> BM25 legs --/
```

The merged candidates still pass through the ONNX cross-encoder reranker and
Pyrrho governance decision.

## Supported Evidence

Foreground ingestion records five kinds of evidence.

| Evidence | Source example | Query expansion |
|---|---|---|
| Parenthetical abbreviation | `Service Level Agreement (SLA)` or `SLA (Service Level Agreement)` | Either form retrieves the other |
| Explicit alias or rename | `Gateway also known as Edge Proxy` | Either stated name can add the other |
| Mechanical identifier form | `AuthService`, `auth_service`, or `auth/service` | A query for `auth service` can add the observed identifier |
| Error context | `error 1722` with `Windows Installer` | The error and sufficiently supported context can expand each other |
| Repeated corpus relationship | `MSI` and `Windows Installer` repeatedly occur together | The related term can enter bounded recall |

The index also records capitalized phrases, acronym tokens, error identifiers,
and repeated two- or three-word phrases as graph nodes. Relationships inferred
from proximity rank below explicit abbreviations, aliases, renames, and
mechanical forms. Weak error context is excluded unless it has adequate weight
or support from more than one document.

Connected high-confidence relationships form collection-local clusters. A
cluster can bridge a query to a related term, but cluster membership receives
the lowest ranking prior. This keeps transitive relationships behind direct
source evidence.

## Safety Boundary

The graph performs mechanical transformations and learns relationships stated
or repeated in the indexed collection. It does not import general-language
synonyms or silently invent private vocabulary.

Safe mechanical forms include:

```text
AuthService  <-> Auth Service
auth_service <-> auth service
auth/service <-> auth service
```

Undocumented domain mappings remain outside the package contract:

```text
Project Falcon <-> authentication rewrite
TC-1001       <-> LOGIN_FAILURE_TEST
Helios        <-> customer portal
```

Those mappings require explicit evidence in source text. Graph terms are recall
suggestions rather than proof that two concepts are universally equivalent.

## Query-Time Flow

1. The deterministic planner extracts overlapping one- to five-token phrases
   exactly from the user's query.
2. The SQLite index matches canonical terms and observed surface forms.
3. Direct forms, supported relationships, and cluster neighbors are ranked.
4. Candidates already present in the query are removed and equivalent
   candidates are de-duplicated.
5. At most six expansion terms are merged into the retrieval plan.
6. The router searches the original query and merged terms as separate BM25
   legs.

Each trace entry includes the expansion term, score, relation type, matched
source form, and supporting-document count. Expansion is bounded and
deterministic for a fixed collection state. If graph expansion fails, retrieval
continues with the prepared literal query plan and records the failure in the
trace.

The ranking order favors direct evidence:

1. abbreviations;
2. explicit aliases and renames;
3. mechanical identifier forms;
4. error context;
5. repeated co-occurrence;
6. cluster membership.

All merged candidates still pass through the configured reranker and Pyrrho
governance.

## Evidence Lifecycle

Semantic facts belong to their source file. Reindexing replaces that file's
facts; deletion retracts them. Aggregate relation weights and clusters are
rebuilt lazily after a corpus mutation, avoiding repeated global work during a
large ingestion run.

Collections created before this feature need to be reindexed before their
existing files contribute graph evidence. Every expanded query retains its
literal retrieval leg.

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
