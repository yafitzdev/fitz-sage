# Configuration Examples

The configuration schema uses:
- **String plugin specs** instead of nested dicts
- **None for disabled** instead of `enabled` flags
- **Sensible defaults** - works out of the box

## Minimal 3-Line Config

The absolute minimum to get started:

```yaml
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small
collection: my_docs
```

Everything else uses sensible defaults:
- Vector DB: pgvector (embedded PostgreSQL via pgserver)
- Reranking: disabled
- Citations: enabled
- Strict grounding: enabled
- Chunk size: 512 tokens

## Common Configurations

### Anthropic + OpenAI (Recommended)

```yaml
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small
collection: my_docs
```

### All Cohere (Single API Key)

```yaml
chat: cohere/command-r-plus
embedding: cohere/embed-english-v3.0
rerank: cohere/rerank-english-v3.0
collection: my_docs
```

### With External PostgreSQL

```yaml
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small
collection: my_docs

vector_db_kwargs:
  mode: external
  connection_string: postgresql://user:pass@localhost:5432/mydb
```

### With Vision/VLM for Images

```yaml
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small
vision: openai/gpt-4o
parser: docling_vision  # Enable VLM parsing
collection: my_docs
```

### Production Setup with All Features

```yaml
# Core plugins
chat: anthropic/claude-sonnet-4
embedding: openai/text-embedding-3-small

# Optional features
rerank: cohere/rerank-english-v3.0
vision: openai/gpt-4o

# Retrieval
retrieval_plugin: dense_rerank
collection: production_docs
top_k: 10

# Generation
enable_citations: true
strict_grounding: true
max_chunks: 12

# Chunking
chunk_size: 768
parser: docling_vision

# Cloud cache
cloud:
  enabled: true
  api_key: ${FITZ_CLOUD_API_KEY}
  org_key: ${FITZ_ORG_KEY}

# External PostgreSQL (for production)
vector_db_kwargs:
  mode: external
  connection_string: ${DATABASE_URL}
```

## Feature Toggle: Enabled vs Disabled

### Reranking

```yaml
# Enabled
rerank: cohere/rerank-english-v3.0

# Disabled
rerank: null
```

### Vision/VLM

```yaml
# Enabled (parse images with VLM)
vision: openai/gpt-4o
parser: docling_vision

# Disabled (figures shown as "[Figure]")
vision: null
parser: docling
```

### Cloud Cache

```yaml
# Enabled
cloud:
  enabled: true
  api_key: ${FITZ_CLOUD_API_KEY}
  org_key: ${FITZ_ORG_KEY}

# Disabled
cloud: null
```

## Plugin String Format

Plugins use a simple string format:

**Format:** `provider` or `provider/model`

**Examples:**
- `cohere` (use provider default model)
- `anthropic/claude-sonnet-4-20250514` (specify model)
- `openai/gpt-4o` (specify model)

**Supported providers:**

| Provider | Example | Notes |
|----------|---------|-------|
| Anthropic | `anthropic/claude-sonnet-4` | Chat + Vision |
| OpenAI | `openai/gpt-4o` | Chat + Embedding + Vision |
| Cohere | `cohere/command-r-plus` | Chat + Embedding + Rerank |
| Ollama | `ollama/llama3` | Local models |
| Azure OpenAI | `azure_openai/gpt-4` | Requires endpoint config |

## Advanced: Plugin Kwargs

For advanced use cases, you can pass additional kwargs to plugins:

```yaml
chat: anthropic/claude-sonnet-4
chat_kwargs:
  temperature: 0.2
  max_tokens: 4096

embedding: openai/text-embedding-3-small
embedding_kwargs:
  dimensions: 1536

rerank: cohere/rerank-english-v3.0
rerank_kwargs:
  top_n: 10
```

## Zero-Config Example

Just use defaults from `default.yaml`:

```python
from fitz_ai.engines.fitz_rag import FitzRagEngine

# Uses all defaults from default_v2.yaml
engine = FitzRagEngine()

# Query immediately
answer = engine.answer(Query("What is quantum computing?"))
```

The only required config value is `collection` (vector DB collection name). Everything else has working defaults.
