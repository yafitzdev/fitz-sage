# Plugin Development Guide

Fitz uses a YAML-based plugin system for LLM providers and vector databases. This guide explains how to create custom plugins.

---

## Feature Control Architecture

**Important:** Optional features (VLM, reranking) are controlled by plugin choice, not config flags.

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG declares WHICH provider/model to use                    │
│  (fitz init configures these sections)                          │
├─────────────────────────────────────────────────────────────────┤
│  vision:                    │  rerank:                          │
│    plugin_name: cohere      │    plugin_name: cohere            │
│    kwargs: {}               │    kwargs:                        │
│                             │      model: rerank-v3.5           │
├─────────────────────────────────────────────────────────────────┤
│  PLUGIN determines IF the feature is used                       │
│  (plugin choice enables/disables the feature)                   │
├─────────────────────────────────────────────────────────────────┤
│  Parser Plugin (VLM control):                                   │
│    docling        → No VLM (figures become "[Figure]")          │
│    docling_vision → Uses VLM from vision: config                │
│                                                                 │
│  Retrieval Plugin (Rerank control):                             │
│    dense          → No reranking (pure vector search)           │
│    dense_rerank   → Uses reranker from rerank: config           │
└─────────────────────────────────────────────────────────────────┘
```

**The pattern:**
- `fitz init` prompts for providers and saves them to config
- Config sections (`vision:`, `rerank:`) specify WHAT provider to use
- Plugin choice specifies IF the feature is used
- No `enabled: true/false` flags - plugin name IS the toggle

**Config locations:**
- Parser: `chunking.default.parser` → `"docling"` or `"docling_vision"`
- Retrieval: `retrieval.plugin_name` → `"dense"` or `"dense_rerank"`

---

## Overview

Fitz uses two types of plugins:

- **YAML plugins** - For external service integrations (LLM providers, vector DBs). Declarative, no code required.
- **Python plugins** - For logic-based components (chunking, parsing, constraints). Require Python code.

**Plugin Types:**

| Type | Format | Location | Purpose |
|------|--------|----------|---------|
| Chat | YAML | `fitz_ai/llm/chat/` | LLM chat/completion |
| Embedding | YAML | `fitz_ai/llm/embedding/` | Text embeddings |
| Rerank | YAML | `fitz_ai/llm/rerank/` | Document reranking |
| Vision | YAML | `fitz_ai/llm/vision/` | VLM for image description |
| Vector DB | YAML | `fitz_ai/vector_db/plugins/` | Vector storage |
| Retrieval | YAML | `fitz_ai/engines/fitz_rag/retrieval/plugins/` | Retrieval strategies |
| Chunking | Python | `fitz_ai/ingestion/chunking/plugins/` | Document chunking |
| Parser | Python | `fitz_ai/ingestion/parser/plugins/` | Document parsing |
| Guardrail | Python | `fitz_ai/core/guardrails/plugins/` | Epistemic safety |

---

## LLM Plugins

### Chat Plugin

Creates a chat/completion provider.

**File:** `fitz_ai/llm/chat/my_provider.yaml`

```yaml
# =============================================================================
# IDENTITY
# =============================================================================
plugin_name: "my_provider"
plugin_type: "chat"
version: "1.0"

# =============================================================================
# PROVIDER
# =============================================================================
provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

# =============================================================================
# AUTHENTICATION
# =============================================================================
auth:
  type: "bearer"                    # bearer, header, query, none
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

# =============================================================================
# ENDPOINT
# =============================================================================
endpoint:
  path: "/chat/completions"
  method: "POST"
  timeout: 120

# =============================================================================
# DEFAULTS
# =============================================================================
# Model tiers for different use cases:
#   smart: Best quality (user-facing queries)
#   fast: Best speed (background tasks, enrichment)
#   balanced: Cost-effective (bulk operations)
defaults:
  models:
    smart: "my-model-large"
    fast: "my-model-small"
    balanced: "my-model-medium"
  temperature: 0.2
  max_tokens: null

# =============================================================================
# REQUEST CONFIGURATION
# =============================================================================
request:
  # Message format transformation
  messages_transform: "openai_chat"   # openai_chat, cohere_chat, anthropic_chat, ollama_chat

  # Static fields added to every request
  static_fields:
    stream: false

  # Map our params to API params
  param_map:
    model: "model"
    temperature: "temperature"
    max_tokens: "max_tokens"

# =============================================================================
# RESPONSE CONFIGURATION
# =============================================================================
response:
  # Path to extract response text (dot notation, array indexing supported)
  content_path: "choices[0].message.content"
  is_array: false
  array_index: 0

  # Optional metadata extraction
  metadata_paths:
    finish_reason: "choices[0].finish_reason"
    tokens_input: "usage.prompt_tokens"
    tokens_output: "usage.completion_tokens"
```

**Message Transforms:**

| Transform | Format |
|-----------|--------|
| `openai_chat` | `{"messages": [{"role": "user", "content": "..."}]}` |
| `cohere_chat` | Cohere v2 format |
| `anthropic_chat` | Anthropic format |
| `ollama_chat` | Ollama format |

---

### Embedding Plugin

Creates an embedding provider.

**File:** `fitz_ai/llm/embedding/my_provider.yaml`

```yaml
plugin_name: "my_provider"
plugin_type: "embedding"
version: "1.0"

provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

auth:
  type: "bearer"
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

endpoint:
  path: "/embeddings"
  method: "POST"
  timeout: 30

defaults:
  model: "my-embed-model"

request:
  # Field name for input text
  input_field: "input"

  # How to wrap input: list, string, object
  input_wrap: "list"

  static_fields: {}

  param_map:
    model: "model"

response:
  # Path to embedding vector(s)
  embeddings_path: "data[0].embedding"
  is_array: true
  array_index: 0
```

---

### Rerank Plugin

Creates a reranking provider.

**File:** `fitz_ai/llm/rerank/my_provider.yaml`

```yaml
plugin_name: "my_provider"
plugin_type: "rerank"
version: "1.0"

provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

auth:
  type: "bearer"
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

endpoint:
  path: "/rerank"
  method: "POST"
  timeout: 30

defaults:
  model: "my-rerank-model"
  top_n: 5

request:
  query_field: "query"
  documents_field: "documents"

  static_fields: {}

  param_map:
    model: "model"
    top_n: "top_n"

response:
  results_path: "results"
  result_index_path: "index"
  result_score_path: "relevance_score"
```

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

See [Unified Storage](features/unified-storage.md) for the full rationale.

### HNSW Index Settings

```yaml
vector_db_kwargs:
  hnsw_m: 16                # Graph connectivity (default: 16)
  hnsw_ef_construction: 64  # Build quality (default: 64)
```

Higher values = better recall but slower indexing.

---

## Using Your Plugin

Once created, LLM plugins are auto-discovered:

```python
from fitz_ai.llm.registry import get_llm_plugin, available_llm_plugins

# List available plugins
print(available_llm_plugins("chat"))      # ['cohere', 'openai', 'my_provider', ...]

# Use your plugin
chat = get_llm_plugin(plugin_name="my_provider", plugin_type="chat")
response = chat.chat([{"role": "user", "content": "Hello"}])

embedder = get_llm_plugin(plugin_name="my_provider", plugin_type="embedding")
vector = embedder.embed("Some text")
```

In configuration:

```yaml
# .fitz/config/fitz_rag.yaml
chat:
  plugin_name: my_provider
  kwargs:
    model: my-model-large

embedding:
  plugin_name: my_provider

# Vector storage (pgvector is the default)
vector_db: pgvector
vector_db_kwargs:
  mode: local
```

---

## Path Notation

Response paths use dot notation with array indexing:

| Path | Accesses |
|------|----------|
| `data` | `response["data"]` |
| `data.text` | `response["data"]["text"]` |
| `data[0]` | `response["data"][0]` |
| `data[0].text` | `response["data"][0]["text"]` |
| `choices[0].message.content` | `response["choices"][0]["message"]["content"]` |

---

## Authentication Types

| Type | Description |
|------|-------------|
| `bearer` | `Authorization: Bearer {key}` |
| `header` | Custom header with key |
| `query` | API key as query parameter |
| `none` | No authentication |

---

## Model Tiers

Chat plugins support three model tiers for cost optimization:

| Tier | Use Case | Example |
|------|----------|---------|
| `smart` | User-facing queries | `gpt-4`, `command-a-03-2025` |
| `fast` | Background tasks, enrichment | `gpt-4o-mini`, `command-r7b` |
| `balanced` | Bulk operations, evaluation | Middle-ground models |

```python
# Use smart tier for queries (default)
chat = get_llm_plugin(plugin_name="cohere", plugin_type="chat", tier="smart")

# Use fast tier for background enrichment
chat = get_llm_plugin(plugin_name="cohere", plugin_type="chat", tier="fast")
```

---

## Existing Plugins

### Chat Plugins

| Plugin | Provider | Models |
|--------|----------|--------|
| `cohere` | Cohere | command-a-03-2025, command-r-plus |
| `openai` | OpenAI | gpt-4o, gpt-4o-mini |
| `anthropic` | Anthropic | claude-3-opus, claude-3-sonnet |
| `azure_openai` | Azure | Deployed OpenAI models |
| `local_ollama` | Ollama | llama3.2, mistral, etc. |

### Embedding Plugins

| Plugin | Provider | Models |
|--------|----------|--------|
| `cohere` | Cohere | embed-english-v3.0 |
| `openai` | OpenAI | text-embedding-3-small |
| `azure_openai` | Azure | Deployed embedding models |
| `local_ollama` | Ollama | nomic-embed-text, etc. |

### Vector DB Plugins

| Plugin | Provider |
|--------|----------|
| `qdrant` | Qdrant (local or cloud) |
| `pinecone` | Pinecone |
| `weaviate` | Weaviate |
| `milvus` | Milvus |
| `local_faiss` | FAISS (local, no server) |

---

## Troubleshooting

### Plugin Not Found

```
LLMRegistryError: Unknown chat plugin: 'my_provider'
```

- Check file is in correct directory (`fitz_ai/llm/chat/my_provider.yaml`)
- Check `plugin_name` matches filename (without `.yaml`)

### Validation Error

```
YAMLPluginValidationError: Invalid YAML plugin
  - provider -> base_url: must start with http://
```

- Check YAML syntax
- Validate against schema requirements
- Use existing plugins as templates

### Authentication Failed

- Verify environment variable is set: `echo $MY_PROVIDER_API_KEY`
- Check `auth.env_vars` lists the correct variable name
- Verify `auth.header_format` matches provider requirements

### Response Extraction Failed

- Test API manually to see actual response structure
- Update `response.content_path` to match actual path
- Use array notation `[0]` if response is wrapped in array
