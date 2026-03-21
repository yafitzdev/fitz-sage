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
│  Configuration (.fitz/config.yaml)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  chat_smart: cohere/command-a-03-2025  ◀─── provider/model string           │
│  chat_fast: cohere/command-r7b-12-2024                                      │
│  embedding: cohere/embed-v4.0                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Factory (fitz_ai/llm/config.py)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  create_chat_provider(spec="cohere/command-a-03-2025")                      │
│                                                                             │
│  Parses provider/model spec, resolves auth, instantiates provider:          │
│  - cohere → CohereChat + ApiKeyAuth(COHERE_API_KEY)                         │
│  - enterprise → EnterpriseChat + CompositeAuth(M2MAuth + ApiKeyAuth)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Provider Instance                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  CohereChat(model="command-a-03-2025", auth=ApiKeyAuth(...))                │
│  - chat(messages) -> str                                                    │
│  - Implements ChatProvider protocol                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Plugin Types

| Type | Format | Purpose | Examples |
|------|--------|---------|----------|
| Chat | Python | LLM completion | Cohere, OpenAI, Anthropic, Ollama, Enterprise |
| Embedding | Python | Vector embeddings | Cohere, OpenAI, Ollama |
| Rerank | Python | Result reranking | Cohere, Ollama |
| Vision | Python | Image understanding | OpenAI, Anthropic, Ollama |
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
│  rerank: cohere/rerank-v3.5  ◀─── Enables reranking (baked in)              │
│  # or                                                                       │
│  rerank: null                ◀─── No reranking                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Feature Control Examples:**

| Feature | Enabled By | Disabled By |
|---------|------------|-------------|
| Reranking | `rerank: cohere/rerank-v3.5` | `rerank: null` (or omit) |
| Vision/VLM | `parser: docling_vision` | `parser: docling` or `parser: glm_ocr` |
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
├── config.yaml           # Main config file
├── pgdata/               # PostgreSQL data (local mode)
└── ingest_state.json     # Incremental ingestion state
```

**Config structure:**

```yaml
# .fitz/config.yaml
chat_fast: cohere/command-r7b-12-2024
chat_balanced: cohere/command-r-08-2024
chat_smart: cohere/command-a-03-2025
embedding: cohere/embed-v4.0
rerank: cohere/rerank-v3.5       # or null to disable
vision: null                     # or cohere (for docling_vision parser)
collection: default
parser: glm_ocr                  # or docling, docling_vision

# Vector storage (PostgreSQL + pgvector)
vector_db: pgvector
vector_db_kwargs:
  mode: local  # or "external" with connection_string
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
│   ├── providers/               # Python providers (Cohere, OpenAI, Anthropic, Ollama, Enterprise)
│   ├── auth/                    # Auth system (ApiKeyAuth, M2MAuth, CompositeAuth)
│   ├── config.py                # Factory dispatch (provider/model → instance)
│   └── client.py                # Public API (get_chat, get_embedder, ...)
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
