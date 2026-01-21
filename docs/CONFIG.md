# Configuration Reference

Complete reference for all Fitz configuration options.

---

## Config File Locations

Fitz uses a two-file configuration structure:

| File | Purpose |
|------|---------|
| `.fitz/config.yaml` | Global config (default engine) |
| `.fitz/config/fitz_rag.yaml` | Engine-specific config |

The `.fitz/` directory is created in your project root when you run `fitz init`.

---

## Global Config

**File:** `.fitz/config.yaml`

```yaml
# Default engine for CLI commands
default_engine: fitz_rag  # Default engine (custom engines can be registered)
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default_engine` | string | `fitz_rag` | Engine used by CLI commands |

---

## Fitz RAG Config

**File:** `.fitz/config/fitz_rag.yaml`

### Complete Example

```yaml
# =============================================================================
# LLM Providers
# =============================================================================

# Chat (LLM for answering questions)
chat:
  plugin_name: cohere          # cohere, openai, anthropic, local_ollama
  kwargs:
    models:
      smart: command-a-03-2025     # Best quality (queries)
      fast: command-r7b-12-2024    # Best speed (enrichment)
      balanced: command-r-08-2024  # Cost-effective (bulk)
    temperature: 0.2

# Embedding (text to vectors)
embedding:
  plugin_name: cohere          # cohere, openai, local_ollama
  kwargs:
    model: embed-english-v3.0

# Reranker (improves retrieval quality)
rerank:
  plugin_name: cohere          # cohere (only provider with rerank)
  kwargs:
    model: rerank-v3.5

# Vision (VLM for describing figures in PDFs)
vision:
  plugin_name: cohere          # cohere, openai, anthropic, local_ollama
  kwargs: {}

# =============================================================================
# Storage
# =============================================================================

# Vector Database
vector_db:
  plugin_name: local_faiss     # local_faiss, qdrant, pinecone, milvus, weaviate
  kwargs: {}
  # For Qdrant:
  # kwargs:
  #   host: localhost
  #   port: 6333

# =============================================================================
# Retrieval
# =============================================================================

retrieval:
  plugin_name: dense           # dense, dense_rerank
  collection: default          # Collection name
  top_k: 5                     # Number of chunks to retrieve

# =============================================================================
# Ingestion
# =============================================================================

# Chunking (document splitting)
chunking:
  default:
    parser: docling            # docling, docling_vision
    plugin_name: recursive     # recursive, simple, markdown, python_code
    kwargs:
      chunk_size: 1000
      chunk_overlap: 200
  by_extension:                # Override by file extension
    .py:
      plugin_name: python_code
    .md:
      plugin_name: markdown

# Enrichment (always on when chat client available)
enrichment:
  enabled: true                  # Master switch
  hierarchy:
    grouping_strategy: metadata  # metadata, semantic
    group_by: source_file

# =============================================================================
# Query Settings
# =============================================================================

# RGS (Retrieval-Guided Synthesis)
rgs:
  enable_citations: true       # Include source citations
  strict_grounding: true       # Only answer from retrieved context
  max_chunks: 8                # Max chunks in context

# =============================================================================
# Logging
# =============================================================================

logging:
  level: INFO                  # DEBUG, INFO, WARNING, ERROR
```

---

## Section Reference

### chat

LLM provider for answering questions.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `cohere` | Provider plugin |
| `kwargs.models.smart` | string | varies | Model for user queries |
| `kwargs.models.fast` | string | varies | Model for background tasks |
| `kwargs.models.balanced` | string | varies | Model for bulk operations |
| `kwargs.temperature` | float | `0.2` | Response randomness (0-1) |

**Available plugins:** `cohere`, `openai`, `anthropic`, `azure_openai`, `local_ollama`

---

### embedding

Embedding provider for text-to-vector conversion.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `cohere` | Provider plugin |
| `kwargs.model` | string | varies | Embedding model name |

**Available plugins:** `cohere`, `openai`, `azure_openai`, `local_ollama`

---

### rerank

Reranking provider for improving retrieval quality.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `cohere` | Provider plugin |
| `kwargs.model` | string | `rerank-v3.5` | Reranking model |

**Available plugins:** `cohere`

**Note:** Reranking is only used when `retrieval.plugin_name: dense_rerank`. See [Feature Control](FEATURE_CONTROL.md).

---

### vision

Vision Language Model for describing figures in PDFs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `cohere` | Provider plugin |
| `kwargs` | object | `{}` | Provider-specific options |

**Available plugins:** `cohere`, `openai`, `anthropic`, `local_ollama`

**Note:** VLM is only used when `chunking.default.parser: docling_vision`. See [Feature Control](FEATURE_CONTROL.md).

---

### vector_db

Vector database for storing and searching embeddings.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `local_faiss` | Database plugin |
| `kwargs` | object | `{}` | Connection options |

**Available plugins:**

| Plugin | Type | kwargs |
|--------|------|--------|
| `local_faiss` | Local | None required |
| `qdrant` | Server | `host`, `port`, `api_key` |
| `pinecone` | Cloud | `api_key`, `environment` |
| `milvus` | Server | `host`, `port` |
| `weaviate` | Server | `host`, `port` |

---

### retrieval

Retrieval strategy for finding relevant chunks.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `plugin_name` | string | `dense` | Retrieval strategy |
| `collection` | string | `default` | Collection to search |
| `top_k` | int | `5` | Chunks to retrieve |

**Available plugins:**

| Plugin | Description |
|--------|-------------|
| `dense` | Pure vector similarity search |
| `dense_rerank` | Vector search + reranking |

---

### chunking

Document chunking configuration.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `default.parser` | string | `docling` | Parser plugin |
| `default.plugin_name` | string | `recursive` | Chunking strategy |
| `default.kwargs.chunk_size` | int | `1000` | Target chunk size (chars) |
| `default.kwargs.chunk_overlap` | int | `200` | Overlap between chunks |
| `by_extension` | object | `{}` | Per-extension overrides |

**Parser plugins:**

| Plugin | Description |
|--------|-------------|
| `docling` | Standard parsing (no VLM) |
| `docling_vision` | VLM-powered figure descriptions |

**Chunking plugins:**

| Plugin | Description |
|--------|-------------|
| `recursive` | General-purpose, respects structure |
| `simple` | Fixed-size chunks |
| `markdown` | Header-aware markdown splitting |
| `python_code` | AST-aware Python splitting |

---

### enrichment

Chunk enrichment pipeline. **All chunk-level enrichment (summary, keywords, entities) is baked in** when a chat client is available. Only hierarchy settings are configurable.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Master switch for all enrichment |

**What's always on (when chat client available):**
- Per-chunk summaries
- Keyword extraction (saved to VocabularyStore)
- Entity extraction

These run via the `ChunkEnricher` bus which batches ~15 chunks per LLM call, making enrichment nearly free.

#### enrichment.hierarchy

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `grouping_strategy` | string | `metadata` | `metadata` or `semantic` |
| `group_by` | string | `source_file` | Metadata key for grouping |
| `n_clusters` | int | null | For semantic grouping |
| `max_clusters` | int | `10` | Max clusters for auto-detect |
| `group_prompt` | string | null | Custom group summary prompt |
| `corpus_prompt` | string | null | Custom corpus summary prompt |

---

### rgs

Retrieval-Guided Synthesis settings for answer generation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_citations` | bool | `true` | Include source citations |
| `strict_grounding` | bool | `true` | Only answer from context |
| `max_chunks` | int | `8` | Max chunks in LLM context |

---

### logging

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `level` | string | `INFO` | Log level |

Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`

---

## Environment Variables

API keys are read from environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| Cohere | `COHERE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| Pinecone | `PINECONE_API_KEY` |

---

## CLI Overrides

Some config values can be overridden via CLI flags:

```bash
# Override collection
fitz ingest ./docs --collection my_docs
fitz query "question" --collection my_docs

# Override top_k
fitz query "question" --top-k 10

# Enable hierarchy at ingest time
fitz ingest ./docs --hierarchy

# Force re-ingestion
fitz ingest ./docs --force
```

---

## See Also

- [Feature Control](FEATURE_CONTROL.md) - How plugins control optional features
- [Plugins](PLUGINS.md) - Plugin development guide
- [CLI](CLI.md) - CLI command reference
