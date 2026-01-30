# Architecture

**Analysis Date:** 2025-01-30

## Pattern Overview

**Overall:** Layered pipeline architecture with unified engine protocol, specialized intelligence subsystems, and clean separation between core contracts and engine-specific implementations.

**Key Characteristics:**
- **Paradigm-agnostic core**: All engines implement `KnowledgeEngine` protocol (`Query → Answer`)
- **Modular pipelines**: RAG pipeline composes retrieval → constraints → context → generation → synthesis
- **Baked-in intelligence**: Query detection, expansion, rewriting, entity linking all automatic (not configurable flags)
- **Plugin-driven configuration**: YAML for LLM/vector DB selection; Python plugins for logic extensibility
- **Clean dependency layering**: Core → Engines, Retrieval, LLM, Storage (no circular imports)

## Layers

### Core Layer
- **Purpose**: Paradigm-agnostic contracts and data models for all engines
- **Location**: `fitz_ai/core/`
- **Contains**: `Query`, `Answer`, `Provenance`, `Constraints`, `KnowledgeEngine` protocol, exceptions, path management, guardrails
- **Depends on**: Nothing (zero external dependencies)
- **Used by**: All engines, CLI, API, SDK

**Key abstractions**:
- `KnowledgeEngine` - Protocol that all engines must implement (single method: `answer(Query) -> Answer`)
- `Query` - Input container with `text`, optional `constraints`, `metadata`
- `Answer` - Output container with `text`, `provenance` (source attribution), `metadata`
- `Provenance` - Source attribution with `source_id`, `excerpt`, `metadata`
- `Constraints` - Query-time epistemic guarantees (detect conflicts, require evidence, etc.)

**Epistemic guardrails** (automatic constraint enforcement):
- `ConflictAwareConstraint` - Detects contradictions between sources
- `InsufficientEvidenceConstraint` - Blocks unsupported claims
- `CausalAttributionConstraint` - Prevents hallucinated causality

### Engine Layer - Fitz RAG
- **Purpose**: Retrieval-augmented generation engine implementation
- **Location**: `fitz_ai/engines/fitz_rag/`
- **Contains**: RAG orchestration, retrieval, generation, answer synthesis, config
- **Depends on**: Core, LLM, Vector DB, Storage, Retrieval Intelligence, Cloud, Structured
- **Used by**: Runtime, CLI, API

**Entry point**:
- `FitzRagEngine` (`fitz_ai/engines/fitz_rag/engine.py`) - Wraps `RAGPipeline` in `KnowledgeEngine` protocol
  - Initializes cloud client if enabled
  - Validates query
  - Runs pipeline
  - Converts pipeline output to `Answer` format

**Pipeline orchestration** (`fitz_ai/engines/fitz_rag/pipeline/engine.py`):
- `RAGPipeline` - Core orchestrator managing retrieval → constraints → routing → context → generation → RGS → synthesis
- `ContextPipeline` - Processes retrieved chunks into generation context
- **Flow**:
  1. Embed query
  2. Run retrieval (with multi-hop for complex questions)
  3. Apply constraint plugins (detect conflicts, validate evidence)
  4. Resolve answer mode (whether to generate with context)
  5. Optionally route to structured query handler
  6. Generate answer using LLM + retrieved context
  7. Synthesize using RGS (Retrieval-Guided Synthesis)

**Components** (`fitz_ai/engines/fitz_rag/pipeline/components.py`):
- `PipelineComponents` - Groups core dependencies (retrieval, chat_factory, RGS)
- `GuardrailComponents` - Epistemic safety (semantic matcher, constraint plugins)
- `RoutingComponents` - Query routing (QueryRouter for intent detection, keyword matcher, multi-hop controller)
- `CloudComponents` - Cache integration (CloudClient, embedder)
- `StructuredComponents` - Structured query handling (schema store, SQL generator, result formatter)

### Retrieval Intelligence Layer
- **Purpose**: Shared query understanding and smart retrieval techniques
- **Location**: `fitz_ai/retrieval/`
- **Contains**: Detection orchestrator, expansion, entity graphs, vocabulary matching, sparse search, rewriting, HyDE
- **Depends on**: Core, LLM (optional for some features)
- **Used by**: Fitz RAG retrieval step, VectorSearchStep

**Detection System** (`fitz_ai/retrieval/detection/`):
- `DetectionOrchestrator` (`detection/registry.py`) - Unified query classification with single LLM call
- `LLMClassifier` (`detection/llm_classifier.py`) - Combines prompt fragments from modules, sends one request to LLM
- `DetectionModule` protocol (`detection/modules/base.py`) - Each module contributes prompt fragment + parsing logic
- **Built-in modules**: Temporal, Aggregation, Comparison, Freshness, Rewriter
- **Dict-based**: ExpansionDetector (synonyms/acronyms)
- **Results**: `DetectionResult[T]` with intent enums (TemporalIntent, AggregationType, etc.)

**Query Routing** (embedded in VectorSearchStep):
- Routes to specialized strategies based on detection:
  - `AggregationSearch` → list/count/enumerate queries
  - `TemporalSearch` → time-based comparisons
  - `ComparisonSearch` → entity comparison queries
  - `SemanticSearch` → standard semantic retrieval

**Supporting subsystems**:
- `entity_graph/` - Links related chunks via shared entities
- `vocabulary/` - Exact-match keyword filtering + expansion
- `sparse/` - BM25 hybrid search with RRF fusion
- `hyde/` - Hypothetical document generation for abstract queries
- `rewriter/` - LLM-based query rewriting (pronouns, compound queries)

### Retrieval Step
- **Location**: `fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py`
- **VectorSearchStep** - Embeds query, applies detection, routes to strategy, applies post-processing (dedup, threshold, rerank, artifact fetch, limit)

### LLM Integration Layer
- **Purpose**: Unified interface to LLM providers (chat, embedding, reranking, vision)
- **Location**: `fitz_ai/llm/`
- **Contains**: Chat factory, embedding client, reranker, provider configs, transforms
- **Depends on**: Core
- **Used by**: Pipeline, detection, retrieval, generation

**Key components**:
- `ChatFactory` - Per-task tier selection (fast/smart/quality)
- `get_embedder()` - Embedding client from config
- `get_reranker()` - Reranker client from config
- Provider configs loaded from YAML

### Vector Database Layer
- **Purpose**: Abstract vector storage with pluggable backends
- **Location**: `fitz_ai/vector_db/`
- **Contains**: Custom client wrapper, pgvector plugin, schemas, loader
- **Depends on**: Core, Storage
- **Used by**: RAG pipeline, ingestion

**Key abstractions**:
- `VectorClient` - Embeddings → vector search + metadata filtering
- `pgvector` plugin (default) - PostgreSQL with pgvector extension
- Vector schemas: chunk embeddings, hierarchy, artifacts

### Storage Layer
- **Purpose**: Database connection management (PostgreSQL)
- **Location**: `fitz_ai/storage/`
- **Contains**: PostgreSQL connection pooling, fitz-pgserver lifecycle management
- **Depends on**: Core
- **Used by**: Vector DB, table store, entity graph, guardrails

**Implementation**:
- Detects if local mode (no connection_string) → launches fitz-pgserver
- Handles crash recovery + stale lock cleanup
- Connection pooling via psycopg_pool

### Cloud Integration Layer
- **Purpose**: Encrypted cache API for RAG answers
- **Location**: `fitz_ai/cloud/`
- **Contains**: CloudClient, AES-256-GCM encryption, cache key generation
- **Depends on**: Core
- **Used by**: RAG pipeline (optional)

**Flow**:
1. Before retrieval: Check cache with `cache_key` (embeddings + query hash)
2. After generation: Store answer in encrypted cache if miss

### Ingestion Pipeline
- **Purpose**: Document → Vector DB workflow (parse → chunk → enrich → embed → store)
- **Location**: `fitz_ai/ingestion/`
- **Contains**: Source detection, parsers, chunkers, enrichment, table handling
- **Depends on**: Core, LLM (for enrichment)
- **Used by**: CLI ingest command

**Flow** (`fitz_ai/ingestion/pipeline/ingestion_pipeline.py`):
1. **File discovery** (`FileScanner`) - Recursive file walk
2. **Parsing** (`ParserRouter`) - Route to appropriate parser (PDF, Markdown, HTML, etc.) → `ParsedDocument`
3. **Chunking** (`ChunkingRouter`) - Route to chunker by file type → `Chunk[]`
4. **Enrichment** (optional) - Summaries, entities, keywords, content type, hierarchy
5. **Embedding** - Embedded by injected embedder
6. **Storage** - VectorDBWriter upserts chunks

**Enrichment System** (`fitz_ai/ingestion/enrichment/`):
- `ChunkEnricher` (bus pattern) - Batches chunks, LLM extracts multiple fields in parallel
- `HierarchyEnricher` - L1 (group summaries) + L2 (corpus summary)
- **Built-in modules**: Summary, Keywords, Entities, ContentType
- **Artifact system** - Auto-generates project-level navigations/catalogs

### Structured Query Handler
- **Purpose**: Route natural language questions to SQL for databases with schemas
- **Location**: `fitz_ai/structured/`
- **Contains**: Schema store, SQL generator, query router, result formatter
- **Depends on**: Core, LLM
- **Used by**: RAG pipeline (optional routing)

### Multi-Hop Reasoning
- **Purpose**: Iterative retrieval for questions requiring multiple steps
- **Location**: `fitz_ai/engines/fitz_rag/retrieval/multihop/`
- **Contains**: HopController, EvidenceEvaluator, BridgeExtractor
- **Flow**: Evaluate → identify missing info → generate bridging query → retrieve again

### Configuration System
- **Purpose**: Layered config loading with deep merge
- **Location**: `fitz_ai/config/` and `fitz_ai/engines/fitz_rag/config/`
- **Config loading** (`config/loader.py`):
  1. Package defaults → `engines/fitz_rag/config/default.yaml`
  2. User overrides → `~/.fitz/config/fitz_rag.yaml`
  3. Deep merge (every key always exists)

**Sections**:
- LLM configs: `chat`, `embedding`, `rerank`, `vision`
- Vector DB: `vector_db` (pgvector default), `vector_db_kwargs`
- Retrieval: `k`, reranking, multi-hop settings
- Chunking: parser choice, chunk size
- Cloud: cache API, encryption keys

### Runtime & Dispatch Layer
- **Purpose**: Single entry point for all engine execution
- **Location**: `fitz_ai/runtime/`
- **Contains**: Engine registry, universal runner
- **Depends on**: Core, all engines

**Entry point**:
- `run()` (`runtime/runner.py`) - Universal query execution
  - Resolves engine factory from registry
  - Loads config (if needed)
  - Creates engine instance
  - Executes query
  - Returns Answer

### CLI Layer
- **Purpose**: User-facing commands
- **Location**: `fitz_ai/cli/`
- **Contains**: 14+ commands (init, ingest, query, config, doctor, etc.)
- **Depends on**: Runtime, core, engines, ingestion
- **Used by**: Users via `fitz` command

### API Layer
- **Purpose**: REST server for programmatic access
- **Location**: `fitz_ai/api/`
- **Contains**: FastAPI routes (query, ingest, collections, health)
- **Depends on**: Runtime, core, engines
- **Runs**: `fitz api` command

### SDK Layer
- **Purpose**: Stateful Python interface for applications
- **Location**: `fitz_ai/sdk/`
- **Contains**: High-level API wrapper
- **Used by**: Python applications, notebooks

## Data Flow

### Query Execution Flow

```
User Input (CLI/API/SDK)
    ↓
Query object (Query { text, constraints, metadata })
    ↓
FitzRagEngine.answer(query)
    ↓
RAGPipeline.run(query_text)
    ├─ 1. Query Embedding: embedder.embed(query) → vector
    │
    ├─ 2. Retrieval Stage:
    │     └─ VectorSearchStep.execute()
    │        ├─ DetectionOrchestrator.detect_for_retrieval(query)
    │        │  ├─ Temporal: time-based queries
    │        │  ├─ Aggregation: list/count queries
    │        │  ├─ Comparison: entity comparison
    │        │  ├─ Freshness: recency/authority
    │        │  └─ Rewriter: pronoun handling
    │        │
    │        ├─ Route to appropriate strategy:
    │        │  ├─ TemporalSearch → temporal index
    │        │  ├─ AggregationSearch → list results
    │        │  ├─ ComparisonSearch → fetch both entities
    │        │  └─ SemanticSearch → vector + sparse + entity expansion
    │        │
    │        ├─ Query Expansion (automatic):
    │        │  ├─ ExpansionDetector: synonyms/acronyms
    │        │  ├─ HyDE: hypothetical docs
    │        │  └─ Multi-query: long queries → multiple focused
    │        │
    │        ├─ Hybrid Search (if sparse available):
    │        │  ├─ Dense: vector similarity
    │        │  ├─ Sparse: BM25 keyword matching
    │        │  └─ Fusion: RRF combine results
    │        │
    │        ├─ Post-processing:
    │        │  ├─ Deduplication (dedupe.py)
    │        │  ├─ Threshold filtering (threshold.py)
    │        │  ├─ Reranking (optional, if enabled)
    │        │  ├─ Artifact fetching (artifact_fetch.py)
    │        │  └─ Limit to k results (limit.py)
    │        │
    │        └─ Return: top-k Chunks with scores
    │
    ├─ 3. Constraint Checking (Epistemic Guardrails):
    │     └─ apply_constraint_plugins(answer, sources)
    │        ├─ ConflictAwareConstraint: detect contradictions
    │        ├─ InsufficientEvidenceConstraint: require evidence
    │        └─ CausalAttributionConstraint: prevent hallucination
    │
    ├─ 4. Answer Mode Resolution:
    │     └─ Determine: generate_with_context or return_chunks_only
    │
    ├─ 5. Multi-Hop Evaluation (if needed):
    │     └─ Is evidence sufficient? Or need additional hops?
    │        ├─ EvidenceEvaluator: assess coverage
    │        ├─ BridgeExtractor: identify gaps
    │        └─ Loop back to retrieval with bridging query
    │
    ├─ 6. Context Processing:
    │     └─ ContextPipeline
    │        ├─ Format chunks for generation
    │        ├─ Apply metadata boosting
    │        └─ Prepare context blocks
    │
    ├─ 7. Generation:
    │     └─ RGS (Retrieval-Guided Synthesis)
    │        ├─ Prompt construction (answer_mode instruction)
    │        ├─ LLM call (chat.complete())
    │        └─ Return: generated text + sources
    │
    └─ Result: RGSAnswer { answer: str, sources: Chunk[], metadata }
         ↓
    Convert to Answer { text, provenance, metadata }
         ↓
    Return to user

```

### Ingestion Data Flow

```
Source Directory
    ↓
FileScanner (discovery)
    ├─ Recursive walk
    └─ Return: FileInfo[]
         ↓
For each file:
    ├─ Check if table file (CSV/TSV)
    │  └─ PostgresTableStore.store() + create schema chunk
    │
    ├─ Parse with ParserRouter
    │  ├─ Route by extension (PDF, MD, HTML, etc.)
    │  └─ Return: ParsedDocument { full_text, metadata, pages }
    │
    ├─ Chunk with ChunkingRouter
    │  ├─ Route by file type
    │  └─ Return: Chunk[] (with source_id, offset, metadata)
    │
    └─ Collect all chunks
         ↓
[Optional] Enrichment Pipeline:
    ├─ ChunkEnricher (bus): batch 15/call
    │  ├─ SummaryModule → chunk.metadata["summary"]
    │  ├─ KeywordModule → VocabularyStore
    │  ├─ EntityModule → chunk.metadata["entities"]
    │  └─ ContentTypeModule → chunk.metadata["content_type"]
    │
    ├─ HierarchyEnricher: L1 group + L2 corpus
    │  └─ Creates additional chunks
    │
    └─ ArtifactRegistry: auto-generate navigation, interfaces, etc.
         ↓
Embedding:
    ├─ vectors = [embedder.embed(get_embedding_text(chunk)) for chunk in chunks]
         ↓
Vector DB Storage:
    └─ VectorDBWriter.upsert(chunks, vectors)

```

## State Management

**Query state**: Immutable - `Query` object passed through pipeline
**Retrieval state**: Chunk list passed between steps with score metadata
**Generation state**: Retrieved chunks + constraints → single Answer
**Session state**: Optional conversation context for multi-turn queries

## Key Abstractions

### KnowledgeEngine Protocol
- **Purpose**: Paradigm-agnostic engine interface
- **Contract**: `def answer(self, query: Query) -> Answer`
- **Example implementations**: `FitzRagEngine`, custom engines
- **Location**: `fitz_ai/core/engine.py`

### DetectionOrchestrator
- **Purpose**: Unified query classification
- **Implementation**: Single LLM call with parallel module prompts
- **Location**: `fitz_ai/retrieval/detection/registry.py`
- **Pattern**: Modules register prompt fragments, orchestrator combines + parses

### EnrichmentBus (ChunkEnricher)
- **Purpose**: Batch LLM enrichment for chunks
- **Pattern**: Modules register extraction logic, bus batches + runs one LLM call
- **Location**: `fitz_ai/ingestion/enrichment/bus.py`

### PipelineComponents
- **Purpose**: Dependency injection for pipeline
- **Pattern**: Group related dependencies, pass as single object
- **Location**: `fitz_ai/engines/fitz_rag/pipeline/components.py`

### RAGPipeline
- **Purpose**: Orchestrate retrieval → constraints → generation
- **Pattern**: Factory method (`from_config()`) + composition
- **Location**: `fitz_ai/engines/fitz_rag/pipeline/engine.py`

## Error Handling

**Strategy**: Explicit error types for actionable recovery

- `QueryError` - User input validation failure
- `KnowledgeError` - Retrieval or knowledge base issue
- `GenerationError` - LLM generation failure
- `ConfigurationError` - Setup/config problem
- `PipelineError` - Internal orchestration failure
- `LLMError` - LLM provider communication
- `RGSGenerationError` - Synthesis failure

**Pattern**: `_wrap_step()` in pipeline - catch all exceptions, raise typed error with context

## Cross-Cutting Concerns

### Logging
- **Framework**: Custom logger wrapper with tags
- **Usage**: `from fitz_ai.logging.logger import get_logger`
- **Tags**: PIPELINE, RETRIEVER, VECTOR_DB, etc. for structured logging

### Configuration Management
- **Layered**: Defaults → user overrides → deep merge
- **Type safety**: Pydantic schemas for all configs
- **Location**: `fitz_ai/config/loader.py`, `fitz_ai/engines/fitz_rag/config/schema.py`

### Path Management
- **Central location**: `fitz_ai/core/paths/`
- **Provides**: Config path, vector DB path, workspace directory
- **Interface**: `FitzPaths` class with class methods

### Instrumentation & Observability
- **Location**: `fitz_ai/core/instrumentation.py`
- **Capabilities**: Tracing, metrics, session logging

---

*Architecture analysis: 2025-01-30*
