# Architecture Overview

High-level system design of Fitz.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User Interface Layer                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │  CLI        │  │  Python SDK │  │  REST API   │                          │
│  │  fitz ...   │  │  import ... │  │  /query     │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
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
│  │  FitzKRAG Engine                │  │  Custom Engines                 │   │
│  │  KRAG pipeline                  │  │  (extensible via registry)      │   │
│  │  - Retrieval                    │  │                                 │   │
│  │  - Constraints (guardrails)     │  │                                 │   │
│  │  - Generation                   │  │                                 │   │
│  └─────────────────────────────────┘  └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  LLM Services       │  │  Storage Layer      │  │  Ingestion Pipeline     │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────────┤
│  - Chat             │  │  PostgreSQL +       │  │  - Parsing              │
│  - Embedding        │  │  pgvector           │  │  - Chunking             │
│  - Rerank           │  │  (vectors, metadata │  │  - Enrichment           │
│  - Vision           │  │   tables, SQL)      │  │  - Embedding            │
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
     ▲              ▲              ▲              ▲              ▲
     │              │              │              │              │
┌──────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────┐
│ llm/     │ │ storage/  │ │ vector_db/│ │retrieval/ │ │ ingestion/ │
│ Chat,    │ │ PostgreSQL│ │ pgvector  │ │ Detection,│ │ Parse,     │
│ Embed,   │ │ connection│ │ abstrac-  │ │ Sparse,   │ │ Chunk,     │
│ Rerank   │ │ manager   │ │ tion      │ │ Entities  │ │ Enrich     │
└──────────┘ └───────────┘ └───────────┘ └───────────┘ └────────────┘
     ▲              ▲              ▲              ▲
     └──────────────┼──────────────┼──────────────┘
                    │              │
                    ┌──────────────────────┐
                    │  engines/            │
                    │  FitzKRAG + custom   │
                    │  Orchestrate layers  │
                    └──────────────────────┘
                               ▲
                    ┌──────────────────────┐
                    │  runtime/            │
                    │  Multi-engine        │
                    │  orchestration       │
                    └──────────────────────┘
                               ▲
                    ┌──────────────────────┐
                    │  cli/, api/, sdk/    │
                    │  User-facing layer   │
                    └──────────────────────┘
```

**Import rules:**

| Layer | Can Import From |
|-------|-----------------|
| `core/` | No imports from engines/, ingestion/ |
| `retrieval/` | `core/` |
| `llm/` | `core/` |
| `storage/` | `core/` |
| `vector_db/` | `core/`, `storage/` |
| `ingestion/` | `core/` |
| `engines/` | `core/`, `llm/`, `vector_db/`, `storage/`, `retrieval/` |
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
              │ Vector DB   │    │ TRUSTWORTHY │    │  Answer +   │
              │ Similarity  │    │ DISPUTED    │    │  Provenance │
              │ Search      │    │ ABSTAIN     │    │  Sources    │
              │             │    │             │    │             │
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
                                │ChunkEnricher│    │ PostgreSQL  │
                                │ + Hierarchy │    │ + pgvector  │
                                │             │    │             │
                                └─────────────┘    └─────────────┘
```

---

## Plugin System

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Configuration (.fitz/config/fitz_krag.yaml)                                │
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
| Vector DB | YAML | Vector storage | pgvector (PostgreSQL) |
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
│  RIGHT: Provider presence                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  rerank: cohere              ◀─── Enables reranking (baked in)              │
│  # or                                                                       │
│  rerank: null                ◀─── No reranking                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Feature Control Examples:**

| Feature | Enabled By | Disabled By |
|---------|------------|-------------|
| Reranking | `rerank: cohere` | `rerank: null` (or omit) |
| Vision/VLM | `parser.plugin_name: docling_vision` | `parser.plugin_name: docling` |
| Enrichment | Chat client available (automatic) | No chat client configured |

---

## Core Types

### Query

```python
@dataclass
class Query:
    text: str                      # The question
    constraints: Constraints = None  # Query-time constraints
    metadata: dict = None          # Additional query metadata
```

### Answer

```python
@dataclass
class Answer:
    text: str                      # The response
    mode: AnswerMode               # TRUSTWORTHY, DISPUTED, ABSTAIN
    provenance: list[Provenance]   # Source attribution chain
    metadata: dict                 # Additional info
```

### Chunk

```python
@dataclass
class Chunk:
    id: str                        # Unique identifier
    content: str                   # Chunk content
    metadata: dict                 # Source file, page, etc.
```

---

## Configuration

```
.fitz/
├── config/
│   └── fitz_krag.yaml    # Main engine config
├── pgdata/               # PostgreSQL data (local mode)
└── ingest_state.json     # Incremental ingestion state
```

**Config structure:**

```yaml
# Engine selection
engine: fitz_krag

# LLM services (YAML plugins)
chat:
  plugin_name: cohere
  kwargs: { model: command-r-plus }

embedding:
  plugin_name: cohere
  kwargs: { model: embed-v4.0 }

# Provider presence enables reranking (no enabled flag)
rerank:
  plugin_name: cohere
  kwargs: { model: rerank-v3.5 }

# Vector storage (PostgreSQL + pgvector)
vector_db: pgvector
vector_db_kwargs:
  mode: local  # or "external" with connection_string

# Ingestion (mixed plugin types)
parser:
  plugin_name: docling_vision    # Python plugin

chunking:
  default:
    plugin_name: semantic        # Python plugin

# Retrieval pipeline
retrieval:
  plugin_name: dense             # YAML plugin

# Enrichment (always on when chat client available)
enrichment:
  hierarchy:
    grouping_strategy: metadata  # or "semantic"
    group_by: source_file
```

---

## Directory Structure

```
fitz_ai/
├── core/                        # Foundation layer
│   ├── types.py                 # Query, Answer, Chunk
│   ├── protocols.py             # KnowledgeEngine protocol
│   └── paths.py                 # Config path management
│
├── engines/                     # Engine implementations
│   └── fitz_krag/
│       ├── engine.py            # Main KRAG engine
│       ├── retrieval/           # Retrieval steps + strategies
│       ├── generation/          # Answer generation + RGS
│       ├── pipeline/            # KRAGPipeline orchestration
│       └── guardrails/plugins/  # Epistemic guardrails (Python)
│
├── retrieval/                   # SHARED retrieval intelligence
│   ├── detection/               # Unified query classification (LLM-based)
│   ├── sparse/                  # BM25 hybrid search
│   ├── entity_graph/            # Entity-based linking
│   ├── vocabulary/              # Keyword storage + matching
│   ├── hyde/                    # Hypothetical document generation
│   └── rewriter/                # LLM-based query rewriting
│
├── llm/                         # LLM service layer
│   ├── chat/                    # Chat plugins (YAML)
│   ├── embedding/               # Embedding plugins (YAML)
│   ├── rerank/                  # Rerank plugins (YAML)
│   └── vision/                  # Vision plugins (YAML)
│
├── storage/                     # PostgreSQL connection manager
│
├── vector_db/                   # Vector DB abstraction
│   └── plugins/                 # DB plugins (YAML)
│
├── ingestion/                   # Document processing
│   ├── parser/plugins/          # Parser plugins (Python)
│   ├── chunking/plugins/        # Chunking plugins (Python)
│   └── enrichment/              # Enrichment pipeline
│
├── cloud/                       # Encrypted cache API
│
├── tabular/                     # CSV/table query with SQL generation
│
├── runtime/                     # Multi-engine orchestration
│
├── cli/                         # CLI commands
│   └── commands/                # Typer commands
│
├── api/                         # REST API (FastAPI)
│
└── sdk/                         # Stateful Python interface
```

---

## Design Principles

1. **Explicit over clever**: No magic. Read the config, know what happens.

2. **Answers over architecture**: Optimize for time-to-insight.

3. **Honest over helpful**: Say "I don't know" rather than hallucinate.

4. **Files over frameworks**: YAML plugins over class hierarchies.

5. **Config-driven**: Provider selection lives only in config files.

6. **Local-first**: Works offline with Ollama + embedded PostgreSQL.

7. **Provenance always**: Every answer traces back to sources.

---

## See Also

- [Unified Storage](features/platform/unified-storage.md) - Why PostgreSQL + pgvector
- [PLUGINS.md](PLUGINS.md) - Plugin development guide
- [CONFIG.md](CONFIG.md) - Configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) - Feature control architecture
- [INGESTION.md](INGESTION.md) - Ingestion pipeline
- [CONSTRAINTS.md](CONSTRAINTS.md) - Epistemic guardrails
