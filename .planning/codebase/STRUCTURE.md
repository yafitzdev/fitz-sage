# Codebase Structure

**Analysis Date:** 2025-01-30

## Directory Layout

```
fitz_ai/
├── core/                           # Paradigm-agnostic protocols & data models
│   ├── engine.py                   # KnowledgeEngine protocol
│   ├── query.py                    # Query input model
│   ├── answer.py                   # Answer output model
│   ├── provenance.py               # Source attribution
│   ├── constraints.py              # Query-time guarantees
│   ├── chunk.py                    # Document chunk model
│   ├── guardrails/                 # Epistemic safety plugins
│   │   ├── base.py                 # ConstraintPlugin ABC
│   │   ├── plugins/                # Built-in constraints
│   │   │   ├── conflict_aware.py
│   │   │   ├── insufficient_evidence.py
│   │   │   └── causal_attribution.py
│   │   ├── semantic.py             # SemanticMatcher for evidence checking
│   │   └── runner.py               # apply_constraint_plugins()
│   ├── paths/                      # Path management (config, vector_db)
│   ├── exceptions.py               # Standard error hierarchy
│   └── instrumentation.py          # Tracing & observability
│
├── engines/
│   └── fitz_rag/                   # Retrieval-augmented generation engine
│       ├── engine.py               # FitzRagEngine (wraps RAGPipeline in KnowledgeEngine)
│       ├── config/
│       │   ├── schema.py           # FitzRagConfig (Pydantic schema)
│       │   ├── default.yaml        # Package defaults
│       │   ├── loader.py           # Config loading (defaults + overrides)
│       │   └── architecture.yaml   # Contract verification config
│       ├── pipeline/
│       │   ├── engine.py           # RAGPipeline (orchestrator)
│       │   ├── components.py       # PipelineComponents (dependency grouping)
│       │   ├── pipeline.py         # ContextPipeline (context formatting)
│       │   └── steps/              # Retrieval pipeline steps
│       │       ├── vector_search.py # VectorSearchStep (intelligent retrieval)
│       │       ├── strategies/      # Query type routing
│       │       │   ├── semantic.py
│       │       │   ├── temporal.py
│       │       │   ├── aggregation.py
│       │       │   └── comparison.py
│       │       ├── rerank.py       # Reranking step
│       │       ├── threshold.py    # Score filtering
│       │       ├── dedupe.py       # Deduplication
│       │       ├── limit.py        # Result limiting
│       │       └── artifact_fetch.py # Fetch generated artifacts
│       ├── retrieval/
│       │   ├── engine.py           # Retrieval orchestrator
│       │   ├── loader.py           # Retrieval plugin loading
│       │   ├── multihop/           # Iterative retrieval for complex questions
│       │   │   ├── controller.py   # HopController (orchestration)
│       │   │   ├── evaluator.py    # EvidenceEvaluator (sufficiency check)
│       │   │   └── bridge.py       # BridgeExtractor (gap identification)
│       │   ├── plugins/            # Retrieval plugins (dense, dense_rerank, etc.)
│       │   └── registry.py         # get_retrieval_plugin()
│       ├── generation/
│       │   ├── answer_mode/        # Answer format instructions
│       │   │   └── instructions.py # get_mode_instruction()
│       │   ├── prompting/          # Prompt construction
│       │   └── retrieval_guided/   # RGS synthesis
│       │       └── synthesis.py    # RGS class + RGSAnswer
│       └── routing/
│           ├── router.py           # QueryRouter (query intent detection)
│           └── models.py           # QueryIntent enum
│
├── retrieval/                      # SHARED intelligence (no engine-specific code)
│   ├── detection/
│   │   ├── protocol.py             # DetectionCategory, DetectionResult, Match
│   │   ├── registry.py             # DetectionOrchestrator (unified orchestrator)
│   │   ├── llm_classifier.py       # LLMClassifier (combines module prompts)
│   │   ├── modules/
│   │   │   ├── base.py             # DetectionModule ABC
│   │   │   ├── temporal.py         # TemporalModule + TemporalIntent enum
│   │   │   ├── aggregation.py      # AggregationModule + AggregationType enum
│   │   │   ├── comparison.py       # ComparisonModule
│   │   │   ├── freshness.py        # FreshnessModule (recency/authority)
│   │   │   └── rewriter.py         # RewriterModule (pronouns, compound)
│   │   └── detectors/
│   │       └── expansion.py        # ExpansionDetector (dict-based synonyms)
│   ├── entity_graph/               # Entity-based chunk linking
│   │   └── store.py                # EntityGraphStore
│   ├── vocabulary/                 # Keyword storage & filtering
│   │   └── store.py                # VocabularyStore + create_matcher
│   ├── sparse/                     # BM25 hybrid search
│   │   └── searcher.py             # BM25Searcher
│   ├── hyde/                       # Hypothetical document generation
│   │   ├── generator.py            # HyDEGenerator
│   │   └── prompts/                # HyDE prompt templates
│   └── rewriter/                   # LLM-based query rewriting
│       ├── rewriter.py             # QueryRewriter
│       └── prompts/                # Rewrite prompts
│
├── ingestion/                      # Document → Vector DB pipeline
│   ├── pipeline/
│   │   └── ingestion_pipeline.py   # IngestionPipeline (orchestrator)
│   ├── source/
│   │   └── base.py                 # SourceFile model
│   ├── parser/
│   │   ├── router.py               # ParserRouter (route by extension)
│   │   └── plugins/                # Parser plugins (docling, pypdf, html, etc.)
│   ├── chunking/
│   │   ├── router.py               # ChunkingRouter (route by file type)
│   │   └── plugins/                # Chunker plugins (semantic, token-based, etc.)
│   ├── enrichment/                 # Batch LLM enrichment (summaries, entities)
│   │   ├── pipeline.py             # EnrichmentPipeline (orchestration)
│   │   ├── bus.py                  # ChunkEnricher (batched LLM bus)
│   │   ├── modules/
│   │   │   ├── base.py             # EnrichmentModule ABC
│   │   │   ├── chunk/              # Per-chunk enrichment
│   │   │   │   ├── summary.py      # SummaryModule
│   │   │   │   ├── keywords.py     # KeywordModule
│   │   │   │   ├── entities.py     # EntityModule
│   │   │   │   └── content_type.py # ContentTypeModule
│   │   │   └── hierarchy/          # Hierarchy enrichment
│   │   │       ├── enricher.py     # HierarchyEnricher (L1/L2 summaries)
│   │   │       ├── grouper.py      # ChunkGrouper
│   │   │       └── matcher.py      # ChunkMatcher
│   │   ├── artifacts/              # Auto-generated project-level artifacts
│   │   │   ├── registry.py         # Artifact plugin system
│   │   │   └── plugins/            # Navigation index, interface catalog, etc.
│   │   ├── prompts/                # Externalized enrichment prompts
│   │   │   ├── chunk/              # Per-module prompts
│   │   │   └── hierarchy/          # Hierarchy prompts
│   │   └── config.py               # Enrichment configuration
│   ├── diff/                       # Change detection for incremental ingestion
│   │   └── scanner.py              # FileScanner (recursive walk)
│   ├── hashing.py                  # Content hashing for deduplication
│   └── detection.py                # Document type detection
│
├── llm/                            # LLM provider abstraction
│   ├── __init__.py                 # get_chat_factory(), get_embedder(), get_reranker()
│   ├── client.py                   # LLM client wrapper
│   ├── factory.py                  # ChatFactory (per-task tier selection)
│   ├── config.py                   # LLM configuration schemas
│   ├── types.py                    # Chat/Embedding/Rerank types
│   ├── auth/                       # Authentication handling
│   ├── providers/                  # Provider implementations
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── cohere.py
│   │   ├── local.py
│   │   └── [other providers]
│   └── transforms.py               # Message/response transforms
│
├── vector_db/                      # Vector database abstraction
│   ├── loader.py                   # VectorClient factory (loads plugins)
│   ├── base.py                     # VectorClient ABC
│   ├── custom.py                   # Custom client wrapper
│   ├── registry.py                 # get_vector_db_plugin()
│   ├── writer.py                   # VectorDBWriter (upsert chunks)
│   ├── types.py                    # Vector search types
│   ├── schemas/                    # Table schemas (chunk embeddings, hierarchy)
│   └── plugins/
│       ├── pgvector.py             # PostgreSQL + pgvector (default)
│       └── [other vector DBs]
│
├── storage/                        # Database connection management
│   ├── postgres.py                 # PostgreSQL pool + pgserver lifecycle
│   └── config.py                   # Storage configuration
│
├── cloud/                          # Encrypted cache API
│   ├── client.py                   # CloudClient (HTTP + encryption)
│   ├── crypto.py                   # AES-256-GCM encryption (org_key stays local)
│   ├── cache_key.py                # Deterministic cache keys
│   └── config.py                   # CloudConfig schema
│
├── config/                         # Global config loading
│   └── loader.py                   # load_engine_config() with layering
│
├── runtime/                        # Universal engine dispatch
│   ├── runner.py                   # run() - single entry point
│   └── registry.py                 # Engine registry & factory
│
├── cli/                            # Command-line interface (14+ commands)
│   ├── commands/
│   │   ├── init.py                 # fitz init
│   │   ├── ingest.py               # fitz ingest
│   │   ├── ingest_runner.py        # Ingestion orchestration
│   │   ├── chat.py                 # fitz chat (interactive)
│   │   ├── query.py                # fitz query (single shot)
│   │   ├── config.py               # fitz config (manage config)
│   │   ├── collections.py          # fitz collections (list, delete)
│   │   ├── doctor.py               # fitz doctor (diagnostics)
│   │   └── [other commands]
│   ├── services/                   # CLI service layer
│   └── ui/                         # UI components (spinners, tables)
│
├── api/                            # FastAPI REST server
│   ├── routes/
│   │   ├── query.py                # POST /query
│   │   ├── ingest.py               # POST /ingest
│   │   ├── collections.py          # GET/DELETE /collections
│   │   ├── health.py               # GET /health
│   │   └── __init__.py
│   ├── models/                     # Pydantic request/response schemas
│   └── main.py                     # FastAPI app setup
│
├── sdk/                            # Stateful Python SDK
│   └── fitz.py                     # FitzSDK class
│
├── tabular/                        # CSV/table support
│   ├── parser.py                   # CSV parsing
│   ├── store/                      # Table storage (PostgreSQL)
│   │   └── base.py                 # TableStore ABC
│   └── models.py                   # Table models
│
├── structured/                     # Structured query handling (optional)
│   ├── router.py                   # QueryRouter (semantic → SQL)
│   ├── executor.py                 # StructuredExecutor (SQL execution)
│   ├── schema_store.py             # SchemaStore (table schemas)
│   ├── sql_generator.py            # SQLGenerator (LLM-based)
│   ├── result_formatter.py         # ResultFormatter
│   └── derived/                    # Derived table management
│
├── backends/                       # Optional backend implementations
│   ├── local_llm/                  # Local LLM support
│   └── local_vector_db/            # Local vector DB alternatives
│
├── prompts/                        # Centralized prompt templates
│   └── [various prompt files]
│
├── plugin_gen/                     # Plugin generation utilities
│
├── logging/                        # Logging infrastructure
│   ├── logger.py                   # get_logger() with tags
│   └── tags.py                     # Standard log tags
│
├── integrations/                   # Third-party integrations
│
└── __init__.py                     # Public API exports (lazy loading)
```

## Directory Purposes

### `fitz_ai/core/`
Everything paradigm-agnostic that all engines use: Query/Answer contracts, exceptions, path management, epistemic guardrails. **No imports from engines or ingestion.**

### `fitz_ai/engines/fitz_rag/`
Complete RAG engine implementation: orchestration, retrieval, generation, synthesis. Specific to RAG paradigm. Uses core + retrieval intelligence.

### `fitz_ai/retrieval/`
**Shared** query understanding and retrieval techniques used by any engine. Detection, expansion, entity graphs, sparse search, rewriting. No engine-specific code.

### `fitz_ai/ingestion/`
Document processing pipeline: discovery → parsing → chunking → enrichment → storage. Used by CLI and background ingestion.

### `fitz_ai/llm/`
LLM provider abstraction. Loads configs from YAML, delegates to providers (OpenAI, Anthropic, Cohere, local). Chat factory for per-task tier selection.

### `fitz_ai/vector_db/`
Vector DB abstraction with pluggable backends. Default is pgvector. Loads schema from YAML config.

### `fitz_ai/storage/`
PostgreSQL connection pooling and lifecycle management. Auto-launches pgserver for local mode. Shared by vector DB, tables, entity graphs.

### `fitz_ai/cloud/`
Encrypted cache API for RAG results. CloudClient handles HTTP + AES-256-GCM encryption (org_key never leaves local machine).

### `fitz_ai/config/`
Global configuration system: layered loading (defaults + user overrides with deep merge). Paths: `~/.fitz/config/`.

### `fitz_ai/runtime/`
Universal engine dispatch. Single `run()` entry point for CLI, API, SDK. Engine registry maps names to factories.

### `fitz_ai/cli/`
14+ user-facing commands for workflow (init, ingest, query, chat, config, doctor, collections, etc.). Commands route through `run()`.

### `fitz_ai/api/`
FastAPI REST server. Routes for query, ingest, collections, health. Also routes through `run()` internally.

### `fitz_ai/sdk/`
Stateful Python wrapper for applications. Single `FitzSDK` class exposing high-level API.

### `fitz_ai/tabular/`
CSV/table support. Parser, storage (PostgreSQL), schema generation. Used during ingestion for table files.

### `fitz_ai/structured/`
Optional structured query handler. Routes natural language to SQL. Schema store, SQL generator, result formatter. Can be disabled.

## Key File Locations

### Entry Points

**Engine entry point:**
- `fitz_ai/engines/fitz_rag/engine.py` - `FitzRagEngine` class

**Universal query entry point:**
- `fitz_ai/runtime/runner.py` - `run(query)` function

**CLI entry point:**
- `fitz_ai/cli/main.py` - Click CLI setup (14+ commands)

**API entry point:**
- `fitz_ai/api/main.py` - FastAPI app

**SDK entry point:**
- `fitz_ai/sdk/fitz.py` - `FitzSDK` class

### Configuration

**Engine config schema:**
- `fitz_ai/engines/fitz_rag/config/schema.py` - `FitzRagConfig` (Pydantic)

**Config loading:**
- `fitz_ai/config/loader.py` - `load_engine_config()`
- `fitz_ai/engines/fitz_rag/config/loader.py` - Engine-specific config loader

**Package defaults:**
- `fitz_ai/engines/fitz_rag/config/default.yaml` - RAG defaults

**User config location:**
- `~/.fitz/config/fitz_rag.yaml` - User overrides

### Core Logic

**RAG orchestration:**
- `fitz_ai/engines/fitz_rag/pipeline/engine.py` - `RAGPipeline` (main orchestrator)

**Retrieval:**
- `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py` - `VectorSearchStep` (intelligent search)
- `fitz_ai/retrieval/detection/registry.py` - `DetectionOrchestrator` (query classification)

**Generation:**
- `fitz_ai/engines/fitz_rag/generation/retrieval_guided/synthesis.py` - `RGS` (Retrieval-Guided Synthesis)

**Enrichment:**
- `fitz_ai/ingestion/enrichment/bus.py` - `ChunkEnricher` (batch enrichment)
- `fitz_ai/ingestion/enrichment/pipeline.py` - `EnrichmentPipeline` (orchestration)

**Ingestion:**
- `fitz_ai/ingestion/pipeline/ingestion_pipeline.py` - `IngestionPipeline` (orchestrator)

### Testing & Validation

**Architecture contracts:**
- `fitz_ai/engines/fitz_rag/config/architecture.yaml` - Dependency rules
- `fitz_ai/engines/fitz_rag/contracts/` - Contract verification

### Guardrails

**Epistemic safety:**
- `fitz_ai/core/guardrails/plugins/` - Constraint implementations
- `fitz_ai/core/guardrails/semantic.py` - `SemanticMatcher` (evidence checking)

## Naming Conventions

### Files

**Patterns observed:**
- `snake_case.py` for all modules
- `_private.py` for internal modules
- `__init__.py` for package exports
- Configuration: `schema.py` (Pydantic), `config.py` (classes), `loader.py` (loading logic)
- Implementations: `{thing}.py` or `{thing}s.py` (e.g., `strategy.py`, `strategies/`)

### Directories

**Patterns observed:**
- `snake_case/` for all directories
- `plugins/` for extensibility points
- `modules/` for plugin-like sub-components
- `prompts/` for templated text
- `config/` for configuration
- Plural for collections: `engines/`, `plugins/`, `modules/`

### Classes

**Patterns observed:**
- `PascalCase` for all classes
- Suffixes indicate role: `...Manager`, `...Engine`, `...Pipeline`, `...Router`, `...Step`, `...Module`, `...Plugin`, `...Store`, `...Factory`, `...Client`
- Protocols: `...Protocol` or no suffix (e.g., `KnowledgeEngine`, `VectorClient`)

### Functions/Methods

**Patterns observed:**
- `snake_case()` for all functions
- `__init__()` for constructors
- `from_config()` for factory methods from config
- `execute()` for pipeline step execution
- `run()` for orchestration/execution
- `get_*()` for factory/retrieval functions
- `create_*()` for factory functions

### Constants

**Patterns observed:**
- `UPPER_SNAKE_CASE` for module constants
- Example: `PIPELINE`, `RETRIEVER`, `VECTOR_DB` (log tags)

## Where to Add New Code

### New Feature (e.g., new retrieval strategy)

**Implementation:**
- Strategy logic: `fitz_ai/retrieval/` or `fitz_ai/engines/fitz_rag/retrieval/`
- Module file: `{feature_name}.py`
- If complex: Create `{feature_name}/` directory with submodules

**Example - Adding temporal search strategy:**
- Location: `fitz_ai/engines/fitz_rag/retrieval/steps/strategies/temporal.py`
- Import in: `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py`
- Route in: `VectorSearchStep.execute()` strategy selection

**Tests:**
- Location: `tests/{parallel_structure}` (mirror source structure)
- Naming: `test_{module}.py` or `test_{class}.py`

### New Detection Module (query understanding)

**Implementation:**
- Module file: `fitz_ai/retrieval/detection/modules/{category}.py`
- Class: Inherit from `DetectionModule`, implement protocol
- Register: Add to `DEFAULT_MODULES` in `fitz_ai/retrieval/detection/modules/__init__.py`

**Prompt:**
- Location: Embedded in module file or externalized to `retrieval/detection/prompts/`
- Return: `DetectionResult` with category, intent enum, matches, metadata

**Tests:**
- Location: `tests/unit/retrieval/detection/modules/test_{category}.py`

### New Enrichment Module (ingestion metadata)

**Implementation:**
- Module file: `fitz_ai/ingestion/enrichment/modules/chunk/{field}.py`
- Class: Inherit from `EnrichmentModule`, implement protocol
- Prompt: `fitz_ai/ingestion/enrichment/prompts/chunk/{field}.txt`
- Register: Add to `create_default_enricher()` in `fitz_ai/ingestion/enrichment/bus.py`

**Return:**
- Module: Apply result via `apply_to_chunk(chunk, result)`
- Storage: `chunk.metadata[{field}]` or external store

**Tests:**
- Location: `tests/unit/ingestion/enrichment/modules/test_{field}.py`

### New LLM Provider

**Implementation:**
- Provider file: `fitz_ai/llm/providers/{provider_name}.py`
- Implement: Chat, Embedder, Reranker clients
- Register: Add to `PROVIDERS` in `fitz_ai/llm/providers/__init__.py`

**Configuration:**
- Schema: Add to `fitz_ai/llm/config.py`
- Env vars: Document in provider module

**Tests:**
- Location: `tests/unit/llm/providers/test_{provider}.py`

### New CLI Command

**Implementation:**
- Command file: `fitz_ai/cli/commands/{command}.py`
- Function: `@click.command()` decorated function
- Register: Add to `fitz_ai/cli/main.py`

**Pattern:**
```python
@click.command()
@click.option("--param", ...)
def {command}(param):
    """Docstring."""
    # Use runtime.run() or pipeline directly
```

**Tests:**
- Location: `tests/integration/cli/test_{command}.py`

### New API Endpoint

**Implementation:**
- Route file: `fitz_ai/api/routes/{resource}.py`
- Schemas: `fitz_ai/api/models/{resource}.py`
- Register: Add to `fitz_ai/api/main.py`

**Pattern:**
```python
@router.post("/{endpoint}")
async def {endpoint}(req: RequestModel) -> ResponseModel:
    """Docstring."""
    # Use runtime.run() or engine directly
```

**Tests:**
- Location: `tests/integration/api/test_{resource}.py`

## Special Directories

### `fitz_ai/prompts/`
- **Purpose**: Centralized prompt templates (optional centralization)
- **Generated**: No
- **Committed**: Yes
- **Organization**: By feature or component

### `fitz_ai/ingestion/enrichment/prompts/`
- **Purpose**: Enrichment module prompts (externalized for maintainability)
- **Generated**: No
- **Committed**: Yes
- **Organization**: `chunk/`, `hierarchy/` subdirectories

### `fitz_ai/retrieval/detection/modules/` and `detectors/`
- **Purpose**: Detection modules (LLM-based) and detectors (dict-based expansion)
- **Generated**: No
- **Committed**: Yes
- **Organization**: By detection category

### `fitz_ai/ingestion/enrichment/artifacts/plugins/`
- **Purpose**: Auto-generated artifact plugins (project-level summaries)
- **Generated**: No
- **Committed**: Yes
- **Organization**: Plugin-per-artifact type

### `~/.fitz/` (User home directory)
- **Purpose**: User workspace (config, collections, vector DB)
- **Generated**: Yes (created by `fitz init`)
- **Committed**: No
- **Organization**:
  - `config/` - Engine configs
  - `collections/` - Collection data
  - `pgserver/` - PostgreSQL data (local mode)

### `.venv/`
- **Purpose**: Python virtual environment
- **Generated**: Yes (created by setup)
- **Committed**: No

## Python Import Organization

**Order (enforced by isort with black profile):**
1. Standard library imports
2. Third-party imports
3. Local imports (from fitz_ai)

**Path aliases:**
- No aliases configured (absolute imports only)

**Example:**
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fitz_ai.core import Query, Answer
from fitz_ai.retrieval.detection import DetectionOrchestrator
```

---

*Structure analysis: 2025-01-30*
