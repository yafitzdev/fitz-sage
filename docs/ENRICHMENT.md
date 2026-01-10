# Enrichment Pipeline

Optional LLM-powered enhancements for ingested content.

---

## Overview

The enrichment pipeline adds AI-generated metadata to chunks during ingestion:

```
┌─────────────────────────────────────────────────────────────────┐
│  Standard Pipeline                                              │
│  Files → Parse → Chunk → Embed → Store                          │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Enrichment Pipeline (optional)                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Summaries  │  │  Entities   │  │  Hierarchy              │  │
│  │             │  │ (GraphRAG)  │  │                         │  │
│  │ Per-chunk   │  │             │  │ Level 0: Chunks         │  │
│  │ descriptions│  │ Extract:    │  │ Level 1: Group summaries│  │
│  │             │  │ - classes   │  │ Level 2: Corpus summary │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Summaries** - Natural language descriptions for better search
- **Entities** - Named entity extraction (GraphRAG only)
- **Hierarchy** - Multi-level summaries for analytical queries

---

## Configuration

Add to your config YAML:

```yaml
# FitzRAG config (entities not available)
enrichment:
  enabled: true
  summary:
    enabled: false       # Per-chunk summaries (expensive!)
  hierarchy:
    enabled: true        # Multi-level summaries
    grouping_strategy: metadata
    group_by: source_file

# GraphRAG config (entities available)
enrichment:
  enabled: true
  entities:
    enabled: true        # Entity extraction
    types: [class, function, api, person, organization, concept]
```

---

## Summaries

LLM-generated descriptions that improve search relevance.

### What it does

For each chunk, generates a natural language summary:

```
Original chunk (code):
  def calculate_refund(order_id, amount):
      """Calculate refund based on return policy."""
      if days_since_purchase(order_id) > 30:
          return 0
      return amount * 0.9  # 10% restocking fee

Generated summary:
  "Refund calculation function that applies a 10% restocking fee
   for returns within 30 days, and denies refunds after 30 days."
```

### When to use

- Dense technical content (code, specs)
- Content where semantic search underperforms
- When you need natural language hooks for retrieval

### Cost warning

**1 LLM call per chunk.** For a codebase with 1000 chunks:
- ~1000 API calls during ingestion
- Significant cost and time

### Configuration

```yaml
enrichment:
  summary:
    enabled: true
    provider: null    # Use default chat provider
    model: null       # Use default model
```

---

## Entities

> **Note:** Entity extraction is for GraphRAG, not FitzRAG. FitzRAG uses keyword vocabulary for exact term matching and will use iterative retrieval for multi-hop queries.

Extracts named entities and domain concepts from chunks.

### What it does

Identifies and extracts:
- **Code entities**: classes, functions, APIs
- **Named entities**: people, organizations
- **Domain concepts**: technical terms, product names

```
Chunk: "The UserService class handles authentication via OAuth2..."

Extracted entities:
  - UserService (class)
  - OAuth2 (api)
  - authentication (concept)
```

### When to use

- Building knowledge graphs
- Entity-based search ("find all mentions of UserService")
- Understanding codebase structure

### Configuration

```yaml
enrichment:
  entities:
    enabled: true
    types:
      - class
      - function
      - api
      - person
      - organization
      - concept
```

### Accessing entities

Entities are stored in chunk metadata:

```python
chunk.metadata["entities"]
# [{"name": "UserService", "type": "class"}, ...]
```

---

## Hierarchy

Multi-level summaries for analytical queries.

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
    └─────────┘          └─────────┘          └─────────┘
```

### How it works

1. **Grouping**: Chunks are grouped (by file, or semantically)
2. **L1 Summaries**: Each group gets an LLM summary
3. **L2 Summary**: All L1 summaries get a corpus summary
4. **Storage**: L1/L2 summaries stored as searchable chunks

### Query behavior

| Query Type | Retrieved |
|------------|-----------|
| "What are the trends?" | L2 corpus + L1 group summaries |
| "Explain the auth module" | L1 auth summary + L0 chunks |
| "How does validateToken work?" | L0 original chunks |

No special query syntax needed - summaries match analytical queries via vector similarity.

### Configuration

```yaml
enrichment:
  hierarchy:
    enabled: true
    grouping_strategy: metadata   # or "semantic"
    group_by: source_file         # metadata key for grouping
    # n_clusters: 10              # for semantic grouping
    # group_prompt: "..."         # custom group summary prompt
    # corpus_prompt: "..."        # custom corpus summary prompt
```

### Grouping strategies

**Metadata grouping** (default):
- Groups by metadata key (e.g., `source_file`)
- Best for: documents, codebases with clear file structure

**Semantic grouping**:
- Clusters chunks by embedding similarity (K-means)
- Best for: mixed content, no clear structure

```yaml
enrichment:
  hierarchy:
    grouping_strategy: semantic
    n_clusters: null        # Auto-detect optimal count
    max_clusters: 10        # Upper bound for auto-detect
```

---

## CLI Usage

### Enable hierarchy at ingest time

```bash
fitz ingest ./docs --hierarchy
```

### Force re-enrichment

```bash
fitz ingest ./docs --force --hierarchy
```

---

## Advanced: Custom Rules

Power users can define custom hierarchy rules:

```yaml
enrichment:
  hierarchy:
    enabled: true
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

## Cost Considerations

| Feature | LLM Calls | When | Engine |
|---------|-----------|------|--------|
| Summaries | 1 per chunk | At ingest | All |
| Entities | 1 per chunk | At ingest | GraphRAG |
| Hierarchy L1 | 1 per group | At ingest | All |
| Hierarchy L2 | 1 total | At ingest | All |

**Example costs (1000 chunks, 10 groups):**

| Feature | Calls | Approx. Cost |
|---------|-------|--------------|
| Summaries only | 1000 | $1-5 |
| Entities only (GraphRAG) | 1000 | $1-5 |
| Hierarchy only | 11 | $0.05-0.10 |

Hierarchy is the most cost-effective enrichment.

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_ai/ingestion/enrichment/config.py` | Configuration schema |
| `fitz_ai/ingestion/enrichment/pipeline.py` | Main orchestrator |
| `fitz_ai/ingestion/enrichment/summary/summarizer.py` | Chunk summaries |
| `fitz_ai/ingestion/enrichment/entities/extractor.py` | Entity extraction |
| `fitz_ai/ingestion/enrichment/hierarchy/enricher.py` | Hierarchy generation |

---

## See Also

- [INGESTION.md](INGESTION.md) - Full ingestion pipeline
- [CONFIG.md](CONFIG.md) - Configuration reference
- [CONSTRAINTS.md](CONSTRAINTS.md) - Epistemic guardrails
