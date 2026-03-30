# Plugin Development Guide

Fitz uses Python providers for LLM services and a plugin system for other components. This guide explains the provider architecture and how to create custom plugins.

---

## Feature Control Architecture

**Important:** Optional features (VLM, reranking) are controlled by plugin choice, not config flags.

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG declares WHICH provider/model to use                    │
│  (edit .fitz/config.yaml)                                       │
├─────────────────────────────────────────────────────────────────┤
│  vision: cohere             │  rerank: cohere/rerank-v3.5      │
│  parser: docling_vision     │                                   │
├─────────────────────────────────────────────────────────────────┤
│  PROVIDER PRESENCE determines IF the feature is used            │
├─────────────────────────────────────────────────────────────────┤
│  Parser (VLM control):                                          │
│    parser: docling        → No VLM (figures become "[Figure]")  │
│    parser: docling_vision → Uses VLM from vision: config        │
│                                                                 │
│  Reranking (Provider-presence control):                         │
│    rerank: null   → No reranking (pure vector search)           │
│    rerank: cohere/rerank-v3.5 → Reranking auto-enabled          │
└─────────────────────────────────────────────────────────────────┘
```

**The pattern:**
- Edit `.fitz/config.yaml` to set providers
- `vision:` and `rerank:` specify WHAT provider/model to use
- VLM: `parser:` choice specifies IF the feature is used
- Reranking: Provider presence enables the feature (baked into `dense` plugin)

**Config locations:**
- Parser: `parser:` → `"docling"`, `"docling_vision"`, or `"glm_ocr"`
- Reranking: `rerank:` → `cohere/rerank-v3.5` (enabled) or `null` (disabled)

---

## Overview

Fitz uses two types of plugins:

- **Python providers** - For LLM service integrations (chat, embedding, rerank, vision). Protocol-based with pluggable auth.
- **Python plugins** - For logic-based components (chunking, parsing, constraints).

**Plugin Types:**

| Type | Format | Location | Purpose |
|------|--------|----------|---------|
| Chat | Python | `fitz_sage/llm/providers/` | LLM chat/completion |
| Embedding | Python | `fitz_sage/llm/providers/` | Text embeddings |
| Rerank | Python | `fitz_sage/llm/providers/` | Document reranking |
| Vision | Python | `fitz_sage/llm/providers/` | VLM for image description |
| Retrieval | YAML | `fitz_sage/engines/fitz_krag/retrieval/plugins/` | Retrieval strategies |
| Chunking | Python | `fitz_sage/ingestion/chunking/plugins/` | Document chunking |
| Parser | Python | `fitz_sage/ingestion/parser/plugins/` | Document parsing |
| Guardrail | Python | `fitz_sage/core/guardrails/plugins/` | Epistemic safety |

**Note:** Vector storage uses PostgreSQL + pgvector (built-in, not pluggable). See [Unified Storage](features/platform/unified-storage.md).

---

## LLM Providers

LLM providers are Python classes that implement protocol interfaces (`ChatProvider`, `EmbeddingProvider`, `RerankProvider`, `VisionProvider`). Each provider wraps a vendor SDK or HTTP client with pluggable authentication.

### Built-in Providers

| Provider | Chat | Embedding | Rerank | Vision |
|----------|------|-----------|--------|--------|
| `cohere` | Yes | Yes | Yes | - |
| `openai` | Yes | Yes | - | Yes |
| `anthropic` | Yes | - | - | Yes |
| `azure_openai` | Yes | Yes | - | Yes |
| `ollama` | Yes | Yes | Yes | Yes |
| `enterprise` | Yes | Yes | - | - |

### Provider Spec Format

Providers are selected using a `provider/model` string:

```python
from fitz_sage.llm import get_chat, get_embedder, get_reranker, get_vision

# Provider only (uses default model for tier)
chat = get_chat("cohere")
chat = get_chat("cohere", tier="fast")

# Provider with explicit model
chat = get_chat("openai/gpt-4o")
embedder = get_embedder("cohere/embed-multilingual-v3.0")
reranker = get_reranker("cohere")
vision = get_vision("openai/gpt-4o")
```

### Provider Protocols

All providers implement `@runtime_checkable` protocols:

```python
class ChatProvider(Protocol):
    def chat(self, messages: list[dict[str, Any]], **kwargs) -> str: ...

class EmbeddingProvider(Protocol):
    def embed(self, text: str, *, task_type: str | None = None) -> list[float]: ...
    def embed_batch(self, texts: list[str], *, task_type: str | None = None) -> list[list[float]]: ...
    @property
    def dimensions(self) -> int: ...

class RerankProvider(Protocol):
    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[RerankResult]: ...

class VisionProvider(Protocol):
    def describe_image(self, image_base64: str, prompt: str | None = None) -> str: ...
```

### Authentication

| Type | Class | Use Case |
|------|-------|----------|
| API Key | `ApiKeyAuth` | Standard providers (OpenAI, Cohere, Anthropic) |
| M2M OAuth2 | `M2MAuth` | Enterprise deployments with client credentials |
| Composite | `CompositeAuth` | Enterprise gateways (M2M + API key combined) |

Auth is resolved automatically from environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| `cohere` | `COHERE_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `azure_openai` | `AZURE_OPENAI_API_KEY` |
| `ollama` | None (no auth) |
| `enterprise` | Configured via auth block |

### Model Tiers

Chat providers support three model tiers for cost optimization:

| Tier | Use Case | Example |
|------|----------|---------|
| `smart` | User-facing queries | `gpt-4o`, `command-a-03-2025` |
| `fast` | Background tasks, enrichment | `gpt-4o-mini`, `command-r7b` |
| `balanced` | Bulk operations, evaluation | Middle-ground models |

### Enterprise Authentication

For enterprise deployments with M2M OAuth2:

```yaml
# In engine config
auth:
  type: enterprise
  token_url: https://auth.example.com/oauth/token
  client_id: ${CLIENT_ID}
  client_secret: ${CLIENT_SECRET}
  llm_api_key_env: LLM_API_KEY
  # Optional mTLS
  cert_path: /path/to/ca.pem
  client_cert_path: /path/to/client.pem
  client_key_path: /path/to/client-key.pem
```

Features: automatic token refresh, exponential backoff, circuit breaker, mTLS support.

---

## Vector DB Plugin

Fitz uses PostgreSQL + pgvector for unified storage of vectors, metadata, and structured tables.

**Plugin:** `pgvector` (default, no configuration needed)

### Configuration

```yaml
# Local mode (default) - embedded PostgreSQL via pgserver
vector_db: pgvector
vector_db_kwargs:
  mode: local

# External mode - your PostgreSQL instance
vector_db: pgvector
vector_db_kwargs:
  mode: external
  connection_string: postgresql://user:pass@host:5432/dbname
```

### Why PostgreSQL?

Fitz uses PostgreSQL + pgvector instead of dedicated vector databases for:

- **Unified storage** - Vectors, metadata, and tables in one database
- **Full SQL** - Real queries, joins, aggregations on structured data
- **Zero friction** - `pip install` includes embedded PostgreSQL (pgserver)
- **One code path** - Same behavior locally and in production

See [Unified Storage](features/platform/unified-storage.md) for the full rationale.

### HNSW Index Settings

```yaml
vector_db_kwargs:
  hnsw_m: 16                # Graph connectivity (default: 16)
  hnsw_ef_construction: 64  # Build quality (default: 64)
```

Higher values = better recall but slower indexing.

---

## Using LLM Providers

In configuration:

```yaml
# .fitz/config.yaml
chat_smart: cohere/command-a-03-2025
chat_fast: cohere/command-r7b-12-2024
embedding: cohere/embed-v4.0
rerank: cohere/rerank-v3.5
collection: default

# Vector storage (pgvector is the default)
vector_db: pgvector
vector_db_kwargs:
  mode: local
```

In code:

```python
from fitz_sage.llm import get_chat, get_embedder, get_reranker, get_vision

chat = get_chat("cohere", tier="smart")
response = chat.chat([{"role": "user", "content": "Hello"}])

embedder = get_embedder("cohere")
vector = embedder.embed("Some text")

reranker = get_reranker("cohere")
results = reranker.rerank("query", ["doc1", "doc2"])
```

---

## Troubleshooting

### Unknown Provider

```
ValueError: Unknown chat provider: 'my_provider'
```

- Check the provider name is one of: `cohere`, `openai`, `anthropic`, `azure_openai`, `ollama`, `enterprise`
- Check the `provider/model` format is correct (e.g., `openai/gpt-4o`)

### Authentication Failed

- Verify environment variable is set: `echo $COHERE_API_KEY`
- Check the correct env var for your provider (see Authentication section above)
- For enterprise auth, verify M2M config fields are complete
