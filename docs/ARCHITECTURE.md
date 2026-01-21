# Architecture Overview

High-level system design of Fitz.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User Interface Layer                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  CLI        │  │  Python SDK │  │  REST API   │  │  fitz() Quickstart  │ │
│  │  fitz ...   │  │  import ... │  │  /query     │  │  One-liner RAG      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Runtime Layer                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Engine Orchestrator                                                │    │
│  │  - Configuration loading                                            │    │
│  │  - Engine instantiation                                             │    │
│  │  - Request routing                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Engine Layer                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐   │
│  │  FitzRAG Engine                 │  │  Custom Engines                 │   │
│  │  Traditional RAG pipeline       │  │  (extensible via registry)      │   │
│  │  - Retrieval                    │  │                                 │   │
│  │  - Constraints (guardrails)     │  │                                 │   │
│  │  - Generation                   │  │                                 │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  LLM Services       │  │  Vector DB          │  │  Ingestion Pipeline     │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────────┤
│  - Chat             │  │  - FAISS (local)    │  │  - Parsing              │
│  - Embedding        │  │  - Pinecone         │  │  - Chunking             │
│  - Rerank           │  │  - Qdrant           │  │  - Enrichment           │
│  - Vision           │  │  - Milvus           │  │  - Embedding            │
└─────────────────────┘  └─────────────────────┘  └─────────────────────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  External Services                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Cohere     │  │  OpenAI     │  │  Anthropic  │  │  Ollama (local)     │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Dependencies

Strict import rules enforce separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│  core/              FOUNDATION - No upward imports              │
│  - Query, Answer    Core data types                             │
│  - Provenance       Source tracking                             │
│  - Protocols        Engine/plugin interfaces                    │
└─────────────────────────────────────────────────────────────────┘
          ▲                    ▲                    ▲
          │                    │                    │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│  llm/           │  │  vector_db/     │  │  ingest/            │
│  Chat, Embed,   │  │  FAISS, Pinecone│  │  Parse, Chunk,      │
│  Rerank, Vision │  │  Qdrant, Milvus │  │  Enrich             │
└─────────────────┘  └─────────────────┘  └─────────────────────┘
          ▲                    ▲                    ▲
          └────────────────────┼────────────────────┘
                               │
                    ┌─────────────────────┐
                    │  engines/           │
                    │  FitzRAG + custom   │
                    │  Orchestrate layers │
                    └─────────────────────┘
                               ▲
                    ┌─────────────────────┐
                    │  runtime/           │
                    │  Multi-engine       │
                    │  orchestration      │
                    └─────────────────────┘
                               ▲
                    ┌─────────────────────┐
                    │  cli/, api/         │
                    │  User-facing layer  │
                    └─────────────────────┘
```

**Import rules:**

| Layer | Can Import From |
|-------|-----------------|
| `core/` | Standard library only |
| `llm/`, `vector_db/`, `ingest/` | `core/` |
| `engines/` | `core/`, `llm/`, `vector_db/` |
| `runtime/` | All layers |
| `cli/`, `api/` | All layers |

Verify with: `python -m tools.contract_map --fail-on-errors`

---

## Data Flow

### Query Flow

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Query  │───▶│  Retrieve   │───▶│ Constraints │───▶│  Generate   │
│         │    │  Chunks     │    │  Check      │    │  Answer     │
└─────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                     │                   │                  │
                     ▼                   ▼                  ▼
              ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
              │ Vector DB   │    │ CONFIDENT   │    │  Answer +   │
              │ Similarity  │    │ QUALIFIED   │    │  Provenance │
              │ Search      │    │ DISPUTED    │    │  Sources    │
              │             │    │ ABSTAIN     │    │             │
              └─────────────┘    └─────────────┘    └─────────────┘
```

### Ingestion Flow

```
┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Files  │───▶│   Parse     │───▶│   Chunk     │───▶│   Embed     │
│         │    │             │    │             │    │             │
└─────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                     │                   │                  │
                     ▼                   ▼                  ▼
              ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
              │ Docling     │    │ Semantic    │    │ Cohere/     │
              │ (PDF, DOCX) │    │ Chunking    │    │ OpenAI/     │
              │ + VLM       │    │ + Metadata  │    │ Ollama      │
              └─────────────┘    └─────────────┘    └─────────────┘
                                       │
                                       ▼
                                ┌─────────────┐    ┌─────────────┐
                                │  Enrich     │───▶│   Store     │
                                │ (always on) │    │             │
                                └─────────────┘    └─────────────┘
                                       │                  │
                                       ▼                  ▼
                                ┌─────────────┐    ┌─────────────┐
                                │ChunkEnricher│    │ FAISS/      │
                                │ + Hierarchy │    │ Pinecone/   │
                                │             │    │ Qdrant      │
                                └─────────────┘    └─────────────┘
```

---

## Plugin System

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Configuration (.fitz/config/fitz_rag.yaml)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  chat:                                                                      │
│    plugin_name: cohere    ◀─── Selects which plugin to use                  │
│    kwargs:                                                                  │
│      model: command-r     ◀─── Passed to plugin constructor                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Registry                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  get_llm_plugin(plugin_type="chat", plugin_name="cohere", **kwargs)         │
│                                                                             │
│  Auto-discovers plugins from:                                               │
│  - fitz_ai/llm/chat/*.yaml                                                  │
│  - fitz_ai/llm/embedding/*.yaml                                             │
│  - etc.                                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Plugin Instance                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  CohereChatPlugin(model="command-r")                                        │
│  - chat(messages) -> str                                                    │
│  - Handles API calls, retries, rate limits                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Plugin Types

| Type | Format | Purpose | Examples |
|------|--------|---------|----------|
| Chat | YAML | LLM completion | Cohere, OpenAI, Anthropic, Ollama |
| Embedding | YAML | Vector embeddings | Cohere, OpenAI, Ollama |
| Rerank | YAML | Result reranking | Cohere |
| Vision | YAML | Image understanding | Cohere, OpenAI |
| Vector DB | YAML | Vector storage | FAISS, Pinecone, Qdrant |
| Retrieval | YAML | Search strategy | Dense, Dense+Rerank |
| Chunking | Python | Text splitting | Semantic, Fixed |
| Parser | Python | Document parsing | Docling, Docling+VLM |
| Guardrail | Python | Epistemic safety | Conflict, Evidence |

---

## Feature Control

Features are controlled by plugin selection, not boolean flags:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WRONG: Boolean flags                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  rerank:                                                                    │
│    enabled: true          ◀─── Anti-pattern                                 │
│    provider: cohere                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  RIGHT: Plugin selection                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  retrieval:                                                                 │
│    plugin_name: dense_rerank   ◀─── Uses reranking                          │
│    # or                                                                     │
│    plugin_name: dense          ◀─── No reranking                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Feature Control Examples:**

| Feature | Enabled By | Disabled By |
|---------|------------|-------------|
| Reranking | `retrieval.plugin_name: dense_rerank` | `retrieval.plugin_name: dense` |
| Vision/VLM | `parser.plugin_name: docling_vision` | `parser.plugin_name: docling` |
| Enrichment | Chat client available (automatic) | `enrichment.enabled: false` |

---

## Core Types

### Query

```python
@dataclass
class Query:
    text: str                      # The question
    collection: str = "default"    # Target collection
    top_k: int = 5                 # Chunks to retrieve
    metadata_filter: dict = None   # Optional filters
```

### Answer

```python
@dataclass
class Answer:
    text: str                      # The response
    mode: AnswerMode               # CONFIDENT, QUALIFIED, DISPUTED, ABSTAIN
    sources: list[Source]          # Provenance chain
    metadata: dict                 # Additional info
```

### Chunk

```python
@dataclass
class Chunk:
    id: str                        # Unique identifier
    text: str                      # Content
    embedding: list[float]         # Vector representation
    metadata: dict                 # Source file, page, etc.
```

---

## Configuration

```
.fitz/
├── config/
│   └── fitz_rag.yaml     # Main engine config
├── vector_db/            # FAISS indices (if local)
└── ingest_state.json     # Incremental ingestion state
```

**Config structure:**

```yaml
# Engine selection
engine: fitz_rag

# LLM services (YAML plugins)
chat:
  plugin_name: cohere
  kwargs: { model: command-r-plus }

embedding:
  plugin_name: cohere
  kwargs: { model: embed-v4.0 }

rerank:
  plugin_name: cohere
  kwargs: { model: rerank-v3.5 }

# Vector storage (YAML plugins)
vector_db:
  plugin_name: local_faiss
  kwargs: { index_type: flat }

# Ingestion (mixed plugin types)
parser:
  plugin_name: docling_vision    # Python plugin

chunking:
  default:
    plugin_name: semantic        # Python plugin

# Features
retrieval:
  plugin_name: dense_rerank      # YAML plugin (controls rerank)

enrichment:
  hierarchy:
    enabled: true                # Direct config (not plugin)
```

---

## Directory Structure

```
fitz_ai/
├── core/                        # Foundation layer
│   ├── types.py                 # Query, Answer, Chunk
│   ├── protocols.py             # KnowledgeEngine protocol
│   ├── paths.py                 # Config path management
│   └── guardrails/plugins/      # Epistemic guardrails (Python)
│
├── engines/                     # Engine implementations
│   └── fitz_rag/
│       ├── engine.py            # Main RAG engine
│       └── retrieval/plugins/   # Retrieval plugins (YAML)
│
├── llm/                         # LLM service layer
│   ├── chat/                    # Chat plugins (YAML)
│   ├── embedding/               # Embedding plugins (YAML)
│   ├── rerank/                  # Rerank plugins (YAML)
│   └── vision/                  # Vision plugins (YAML)
│
├── vector_db/                   # Vector storage
│   └── plugins/                 # DB plugins (YAML)
│
├── ingestion/                   # Document processing
│   ├── parser/plugins/          # Parser plugins (Python)
│   ├── chunking/plugins/        # Chunking plugins (Python)
│   └── enrichment/              # Enrichment pipeline
│
├── cli/                         # CLI commands
│   └── commands/                # Typer commands
│
└── api/                         # REST API
    └── routes/                  # FastAPI routes
```

---

## Design Principles

1. **Explicit over clever**: No magic. Read the config, know what happens.

2. **Answers over architecture**: Optimize for time-to-insight.

3. **Honest over helpful**: Say "I don't know" rather than hallucinate.

4. **Files over frameworks**: YAML plugins over class hierarchies.

5. **Config-driven**: Provider selection lives only in config files.

6. **Local-first**: Works offline with Ollama + FAISS.

7. **Provenance always**: Every answer traces back to sources.

---

## See Also

- [PLUGINS.md](PLUGINS.md) - Plugin development guide
- [CONFIG.md](CONFIG.md) - Configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) - Feature control architecture
- [INGESTION.md](INGESTION.md) - Ingestion pipeline
- [CONSTRAINTS.md](CONSTRAINTS.md) - Epistemic guardrails
