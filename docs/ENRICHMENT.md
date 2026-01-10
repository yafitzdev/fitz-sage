# Enrichment Pipeline

LLM-powered enhancements for ingested content. **Always on, nearly free.**

---

## Overview

The enrichment pipeline adds AI-generated metadata to chunks during ingestion. All enrichment is **baked in** - no configuration needed.

```
┌─────────────────────────────────────────────────────────────────┐
│  Standard Pipeline                                              │
│  Files → Parse → Chunk → Embed → Store                          │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Enrichment Pipeline (always on)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              ChunkEnricher (Enrichment Bus)             │    │
│  │         One LLM call per batch (~15 chunks)             │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────────────┐    │    │
│  │  │  Summary  │  │ Keywords  │  │     Entities      │    │    │
│  │  │  Module   │  │  Module   │  │      Module       │    │    │
│  │  │           │  │           │  │                   │    │    │
│  │  │ Per-chunk │  │ Exact-    │  │ Named entities    │    │    │
│  │  │ search    │  │ match IDs │  │ (class, person,   │    │    │
│  │  │ summaries │  │ for vocab │  │  technology...)   │    │    │
│  │  └───────────┘  └───────────┘  └───────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                   Hierarchy Enricher                   │      │
│  │                                                        │      │
│  │  Level 0: Chunks (with enrichments from above)         │      │
│  │  Level 1: Group summaries (per source file)            │      │
│  │  Level 2: Corpus summary (all documents)               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Summaries** - Natural language descriptions for better semantic search
- **Keywords** - Exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- **Entities** - Named entity extraction (classes, people, technologies)
- **Hierarchy** - Multi-level summaries for analytical queries

All features run automatically when a chat client is available.

---

## Architecture: ChunkEnricher

The `ChunkEnricher` is the unified enrichment bus that extracts multiple enrichments in a single LLM call per batch of chunks.

### How it works

1. **Batching**: Chunks are processed in batches of ~15
2. **Single LLM call**: One call extracts summary + keywords + entities for the entire batch
3. **JSON response**: LLM returns structured JSON with all enrichments
4. **Module parsing**: Each module parses its portion and applies to chunks

```python
# One LLM call returns:
[
  {
    "summary": "Authentication module handling OAuth2 flows...",
    "keywords": ["AuthService", "OAuth2", "JWT_TOKEN"],
    "entities": [{"name": "AuthService", "type": "class"}, ...]
  },
  # ... for each chunk in batch
]
```

### Extensibility

Adding new enrichment types is simple - implement `EnrichmentModule`:

```python
class EnrichmentModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def json_key(self) -> str: ...

    @abstractmethod
    def prompt_instruction(self) -> str: ...

    @abstractmethod
    def parse_result(self, data: Any) -> Any: ...

    def apply_to_chunk(self, chunk: Chunk, result: Any) -> None:
        pass  # Override to attach to chunks
```

New modules ride the same LLM call at zero additional cost.

---

## Built-in Modules

### Summary Module

Generates searchable descriptions for each chunk.

**Input:**
```python
def calculate_refund(order_id, amount):
    """Calculate refund based on return policy."""
    if days_since_purchase(order_id) > 30:
        return 0
    return amount * 0.9  # 10% restocking fee
```

**Output:**
```
"Refund calculation function that applies a 10% restocking fee
 for returns within 30 days, and denies refunds after 30 days."
```

**Stored in:** `chunk.metadata["summary"]`

---

### Keyword Module

Extracts exact-match identifiers for vocabulary-based retrieval.

**What it extracts:**
- Test cases: `TC-1001`, `testcase_42`
- Tickets: `JIRA-4521`, `BUG-789`
- Versions: `v2.0.1`, `1.0.0-beta`
- Code identifiers: `AuthService`, `handle_login()`
- Constants: `MAX_RETRIES`, `API_KEY`
- API endpoints: `/api/v2/users`

**Stored in:** `VocabularyStore` for exact-match retrieval at query time.

---

### Entity Module

Extracts named entities and domain concepts.

**Entity types:**
- `class` - Code classes
- `function` - Functions/methods
- `person` - People mentioned
- `organization` - Companies, teams
- `technology` - Tools, frameworks, protocols
- `concept` - Domain concepts

**Stored in:** `chunk.metadata["entities"]`

---

## Hierarchy

Multi-level summaries for analytical queries. **Always on by default.**

### The problem

Standard RAG struggles with analytical questions:

```
Q: "What are the main themes in these documents?"
Standard RAG: Returns random individual chunks (not useful)
```

### The solution

Hierarchy creates summary layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 2: Corpus Summary                                        │
│  "Across 50 documents, the main themes are: API design,         │
│   security best practices, and deployment patterns."            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Level 1: Group  │  │ Level 1: Group  │  │ Level 1: Group  │
│ "auth/*.py:     │  │ "api/*.py:      │  │ "deploy/*.py:   │
│  OAuth2 impl,   │  │  REST endpoints,│  │  Docker config, │
│  JWT handling"  │  │  validation"    │  │  K8s manifests" │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         ▲                    ▲                    ▲
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │ Level 0 │          │ Level 0 │          │ Level 0 │
    │ Chunks  │          │ Chunks  │          │ Chunks  │
    │ (with   │          │ (with   │          │ (with   │
    │summary, │          │summary, │          │summary, │
    │keywords,│          │keywords,│          │keywords,│
    │entities)│          │entities)│          │entities)│
    └─────────┘          └─────────┘          └─────────┘
```

### Query behavior

| Query Type | Retrieved |
|------------|-----------|
| "What are the trends?" | L2 corpus + L1 group summaries |
| "Explain the auth module" | L1 auth summary + L0 chunks |
| "How does validateToken work?" | L0 original chunks |

No special query syntax needed - summaries match analytical queries via vector similarity.

### Configuration

Hierarchy configuration is the only enrichment setting available:

```yaml
enrichment:
  enabled: true  # Master switch (default: true)
  hierarchy:
    grouping_strategy: metadata   # or "semantic"
    group_by: source_file         # metadata key for grouping
    # n_clusters: 10              # for semantic grouping
    # group_prompt: "..."         # custom group summary prompt
    # corpus_prompt: "..."        # custom corpus summary prompt
```

---

## CLI Usage

Enrichment runs automatically on every ingestion. No flags needed:

```bash
fitz ingest ./docs
```

### Force re-enrichment

To regenerate enrichments for already-ingested files:

```bash
fitz ingest ./docs --force
```

---

## Cost Analysis

The ChunkEnricher's batching makes enrichment **nearly free**.

### Per batch (~15 chunks)

| Component | Tokens |
|-----------|--------|
| Prompt overhead | ~500 |
| Chunk content (15 × ~400) | ~6,000 |
| **Total input** | **~6,500** |
| Response (15 × ~100) | ~1,500 |
| **Total output** | **~1,500** |

### Cost per model

| Model | Per Batch | Per Chunk | 1000 Chunks |
|-------|-----------|-----------|-------------|
| Claude 3.5 Haiku | $0.011 | $0.0007 | **$0.74** |
| GPT-4o-mini | $0.002 | $0.0001 | **$0.13** |

**For under $1, you get summary + keywords + entities for your entire codebase.**

Adding new enrichment modules costs nothing extra - they ride the same LLM call.

### Comparison with old architecture

| Architecture | LLM Calls (1000 chunks) | Cost |
|--------------|-------------------------|------|
| Old: 1 call per chunk | 1,000 | $1-5 |
| **New: Batched ChunkEnricher** | **67** | **$0.13-0.74** |

---

## Advanced: Custom Rules

Power users can define custom hierarchy rules:

```yaml
enrichment:
  hierarchy:
    rules:
      - name: video_comments
        paths: ["comments/**/*.txt"]
        group_by: video_id
        prompt: "Summarize the sentiment and key themes in these comments."
        corpus_prompt: "What are the overall trends across all videos?"

      - name: support_tickets
        paths: ["tickets/**/*.json"]
        group_by: category
        prompt: "Summarize the main issues in this category."
```

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_ai/ingestion/enrichment/chunk/enricher.py` | ChunkEnricher bus and modules |
| `fitz_ai/ingestion/enrichment/pipeline.py` | Main orchestrator |
| `fitz_ai/ingestion/enrichment/config.py` | Configuration schema |
| `fitz_ai/ingestion/enrichment/hierarchy/enricher.py` | Hierarchy generation |
| `fitz_ai/ingestion/vocabulary/store.py` | Keyword vocabulary storage |

---

## See Also

- [INGESTION.md](INGESTION.md) - Full ingestion pipeline
- [CONFIG.md](CONFIG.md) - Configuration reference
- [CONSTRAINTS.md](CONSTRAINTS.md) - Epistemic guardrails
