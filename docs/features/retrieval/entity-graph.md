# Entity Graph (Related Chunk Discovery)

## Problem

Standard semantic search retrieves chunks independently. This fails when:

- **Q:** "What else mentions TechCorp?"
- **Standard RAG:** Only returns chunks that match the query semantically
- **Expected:** Also return chunks that mention the same entities (TechCorp, its products, people, etc.)

Semantic similarity doesn't capture entity relationships. If Chunk A and Chunk B both mention "AuthService" but discuss different aspects, they won't be retrieved together unless the query happens to match both.

## Solution: Entity-Based Chunk Linking

Build a graph linking entities to chunks during ingestion. At query time, expand retrieved chunks by finding related chunks via shared entities.

```
Initial retrieval:     [Chunk A (mentions AuthService, OAuth2)]
                              ↓
                    Entity Graph Lookup
                              ↓
                    Shared entities: AuthService, OAuth2
                              ↓
                    Find chunks mentioning same entities
                              ↓
Expanded results:      [Chunk A, Chunk B (AuthService), Chunk C (OAuth2)]
```

## How It Works

### At Ingestion Time

1. **Entity extraction** - The EntityModule (part of ChunkEnricher) extracts named entities from each chunk
2. **Graph population** - Entities and chunk associations stored in PostgreSQL:
   - `entities` table: Entity names, types, mention counts
   - `entity_chunks` table: Many-to-many mapping of entities to chunks

```
Chunk: "The AuthService class handles OAuth2 authentication..."
          ↓
Entities: [("AuthService", "class"), ("OAuth2", "technology")]
          ↓
Graph edges:  AuthService → chunk_abc123
              OAuth2 → chunk_abc123
```

### At Query Time

1. **Initial retrieval** - Standard semantic search returns top-k chunks
2. **Entity lookup** - Get entities mentioned in retrieved chunks
3. **Graph expansion** - Find other chunks sharing those entities
4. **Ranking** - Rank related chunks by number of shared entities
5. **Merge** - Combine with original results (avoiding duplicates)

```python
# Simplified flow in VectorSearchStep
initial_chunks = semantic_search(query, k=5)
entity_names = get_entities_for_chunks(initial_chunks)
related_chunks = entity_graph.get_related_chunks(
    chunk_ids=initial_chunks,
    max_total=10,
    min_shared_entities=2
)
final_chunks = merge_and_dedupe(initial_chunks, related_chunks)
```

## Key Design Decisions

1. **Always-on** - Baked into the enrichment and retrieval pipelines. No configuration.

2. **PostgreSQL storage** - Uses the unified storage system. No separate graph database.

3. **Lightweight graph** - Only stores entity-chunk edges, not full entity attributes.

4. **Ingestion-time extraction** - Entities extracted once during ingestion (via LLM), not at query time.

5. **Configurable expansion** - `min_shared_entities` controls how related chunks must be (default: 1).

6. **Corpus summary injection** - For thematic queries, entity graph expansion also injects corpus-level summaries to provide broader context beyond individual chunk matches.

## Entity Types

The EntityModule extracts these entity types:

| Type | Examples |
|------|----------|
| `class` | AuthService, UserController, PaymentGateway |
| `function` | validateToken(), processPayment() |
| `person` | John Smith, Alice (when mentioned as people) |
| `organization` | TechCorp, Acme Inc, Engineering Team |
| `technology` | OAuth2, PostgreSQL, React, Kubernetes |
| `concept` | Authentication, Rate Limiting, Caching |

## Configuration

No configuration required. Feature is baked into the enrichment and retrieval pipelines.

Internal parameters:
- `max_total`: Maximum related chunks to retrieve (default: 20)
- `min_shared_entities`: Minimum shared entities to consider related (default: 1)

## Files

- **Graph store:** `fitz_ai/retrieval/entity_graph/store.py`
- **Entity extraction:** `fitz_ai/ingestion/enrichment/modules/chunk/entities.py`
- **Integration:** `fitz_ai/engines/fitz_krag/retrieval/steps/vector_search.py`

## Benefits

| Without Entity Graph | With Entity Graph |
|---------------------|-------------------|
| Only semantically similar chunks | Also conceptually related chunks |
| Miss chunks about same entities | Discover via shared entity links |
| No entity-based exploration | "What else mentions X?" works |
| Isolated chunk retrieval | Connected knowledge retrieval |

## Example

**Documents:**
- Chunk A: "The AuthService class validates JWT tokens using OAuth2 protocol."
- Chunk B: "AuthService logs all authentication attempts to the audit table."
- Chunk C: "OAuth2 refresh tokens expire after 30 days by default."

**Query:** "How does authentication work?"

**Without Entity Graph:**
- Returns: Chunk A (best semantic match)

**With Entity Graph:**
- Initial: Chunk A (semantic match)
- Entities found: AuthService, OAuth2
- Graph expansion: Chunk B (shares AuthService), Chunk C (shares OAuth2)
- Returns: Chunk A, Chunk B, Chunk C

The LLM now has complete context about AuthService behavior and OAuth2 configuration.

## Dependencies

- Requires EntityModule in enrichment pipeline (always on)
- PostgreSQL for graph storage (unified storage)
- Part of VectorSearchStep (no additional latency for graph lookup)

## Related Features

- **Enrichment** - EntityModule extracts entities during ingestion
- **Multi-Hop Reasoning** - Can use entity graph for traversal hints
- **Comparison Queries** - Entity graph helps retrieve both compared entities
