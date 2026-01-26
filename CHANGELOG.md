# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.7.0] - 2026-01-26

### 🎉 Highlights

**Unified PostgreSQL Storage** - Replaced FAISS + SQLite with PostgreSQL + pgvector for all storage needs. Uses `pgserver` (pip-installable embedded PostgreSQL) for local mode with zero external dependencies. One database per collection, automatic schema management, and HNSW indexing for fast vector search.

**Native PostgreSQL Tables** - Tabular data (CSV/tables) now stored directly in PostgreSQL with automatic schema inference. Enables SQL queries over structured data alongside vector search.

**LLM Factory Pattern** - New `chat_factory()` function provides clean LLM client instantiation with proper dependency injection. Replaces scattered client creation logic.

### 🚀 Added

#### PostgreSQL Storage System (`fitz_ai/storage/`)
- `PostgresConnectionManager` - Singleton connection manager with pgserver lifecycle
- `StorageConfig` - Pydantic config for local/external PostgreSQL modes
- Per-collection database isolation (one DB per collection)
- Automatic pgvector extension initialization
- Connection pooling via `psycopg_pool`
- pgserver graceful shutdown with file handle cleanup on Windows
- Auto-recovery on corrupted data directories

#### pgvector Backend (`fitz_ai/backends/local_vector_db/pgvector.py`)
- `PgVectorDB` - Full VectorDBPlugin implementation
- HNSW indexing with configurable `m` and `ef_construction`
- Hybrid search combining vector similarity + full-text search (tsvector)
- Native PostgreSQL `tsvector` for sparse/BM25-style retrieval
- `scroll()` and `scroll_with_vectors()` for batch iteration
- Automatic schema creation on first use

#### PostgreSQL Table Store (`fitz_ai/tabular/store/postgres.py`)
- `PostgresTableStore` - Native PostgreSQL storage for tabular data
- Gzip-compressed CSV storage in BYTEA column
- Hash-based deduplication
- Automatic schema inference and column tracking

#### LLM Factory (`fitz_ai/llm/chat/factory.py`)
- `chat_factory()` - Clean factory function for chat client instantiation
- Proper dependency injection pattern
- Tier-based model selection (smart/fast)

#### Test Infrastructure
- Test tier markers: `tier1` (unit), `tier2` (integration), `tier3` (e2e), `tier4` (performance)
- `pytest.ini` configuration for tier-based test execution
- pgserver test fixtures with auto-cleanup
- Windows-specific file handle cleanup in tests

### 🔄 Changed

- **Default vector DB**: `faiss` → `pgvector` in default config
- **Vocabulary storage**: YAML files → PostgreSQL `keywords` table
- **Sparse index**: BM25 files → PostgreSQL `tsvector` column (auto-maintained)
- **Entity graph**: SQLite files → PostgreSQL `entities` + `entity_chunks` tables
- **Table store**: SQLite/generic → PostgreSQL native tables
- **Collection delete**: Now drops entire PostgreSQL database (auto-cleans all related data)

### 🗑️ Removed

#### Legacy Storage Backends
- `fitz_ai/backends/local_vector_db/faiss.py` - Replaced by pgvector
- `fitz_ai/vector_db/plugins/local_faiss.yaml` - Replaced by pgvector.yaml
- `fitz_ai/tabular/store/sqlite.py` - Replaced by postgres.py
- `fitz_ai/tabular/store/generic.py` - Replaced by postgres.py
- `fitz_ai/tabular/store/qdrant.py` - Replaced by postgres.py
- `fitz_ai/tabular/store/cache.py` - No longer needed

#### Knowledge Map Module
- `fitz_ai/map/` - Experimental module removed (not production-ready)
- `fitz map` CLI command removed

#### Deprecated Path Helpers
- `vocabulary()` path function - Now emits deprecation warning
- `sparse_index()` path function - Now emits deprecation warning
- `entity_graph()` path function - Now emits deprecation warning

### 📦 Dependencies

New:
```toml
"psycopg[binary]>=3.1"
"psycopg-pool>=3.1"
"pgvector>=0.2.0"
"pgserver>=0.1.0"
```

Removed:
```toml
"faiss-cpu>=1.7.0"  # Now optional via [faiss] extra
```

### 🧪 Testing

- 198 tier1 tests passing
- 62 postgres-specific tests
- 67 vocabulary tests (migrated to PostgreSQL)
- Windows-compatible pgserver tests with file handle cleanup

---

## [0.6.2] - 2026-01-24

### 🎉 Highlights

**Unified LLM-Based Query Classification** - Consolidated all scattered query detection systems (temporal, aggregation, comparison, freshness, expansion) into a single unified detection module. One LLM call now classifies all query intents instead of separate regex-based detectors. More accurate classification with lower latency.

### 🚀 Added

#### Unified Detection System (`fitz_ai/retrieval/detection/`)
- `LLMClassifier` - Combines all detection modules into one LLM call
- `DetectionOrchestrator` - Unified registry with `DetectionSummary` result
- `DetectionModule` ABC - Modular detection with `prompt_fragment()` and `parse_result()`
- Detection modules: `TemporalModule`, `AggregationModule`, `ComparisonModule`, `FreshnessModule`, `RewriterModule`
- `ExpansionDetector` - Dict-based synonym/acronym expansion (non-LLM, fast)
- `DetectionResult` dataclass with confidence scores and metadata
- `DetectionCategory` enum for type-safe detection types

### 🔄 Changed

- **Query classification is now LLM-based** - More accurate than regex patterns, handles edge cases better
- **VectorSearchStep** - Now uses `DetectionOrchestrator` for all query classification
- **Retrieval strategies** - Updated to receive `DetectionResult` instead of legacy detector outputs

### 🗑️ Removed

#### Legacy Detection Systems (consolidated into unified detection)
- `fitz_ai/retrieval/aggregation/` - Replaced by `detection/modules/aggregation.py`
- `fitz_ai/retrieval/temporal/` - Replaced by `detection/modules/temporal.py`
- `fitz_ai/retrieval/expansion/` - Replaced by `detection/detectors/expansion.py`
- `fitz_ai/engines/fitz_rag/retrieval/steps/freshness.py` - Replaced by `detection/modules/freshness.py`

### 📚 Documentation

- Updated feature docs to reflect unified detection system
- CLAUDE.md already documented the new detection architecture

---

## [0.6.1] - 2026-01-23

### 🎉 Highlights

**HyDE (Hypothetical Document Embeddings)** - Generate hypothetical document passages that would answer abstract queries, then search with both original and hypothetical embeddings. Bridges the semantic gap between conceptual questions and concrete document content. Queries like "What's their approach to sustainability?" now find relevant EV/battery/emissions docs.

**Contextual Embeddings** - Chunks are now embedded with their summaries prepended, providing richer semantic context. This resolves pronoun ambiguity and improves retrieval quality for chunks that reference concepts without naming them explicitly. Inspired by Anthropic's Contextual Retrieval technique.

**Query Rewriting** - LLM-powered query rewriting resolves conversational context (pronouns, references), fixes typos, removes filler words, and optimizes queries for document retrieval. Enables natural multi-turn conversations with proper context resolution.

**Conversational Context for SDK & API** - The SDK and REST API now support passing conversation history for context-aware retrieval. Queries like "tell me more about it" now work correctly in programmatic use cases.

### 🚀 Added

#### HyDE - Hypothetical Document Embeddings (`fitz_ai/retrieval/hyde/`)
- `HypothesisGenerator` - Generates 3 hypothetical document passages per query
- Single fast-tier LLM call for all hypotheses
- Hypotheses embedded and searched alongside original query
- RRF (Reciprocal Rank Fusion) merges results from all searches
- Always-on when chat client is available (no configuration needed)
- Graceful degradation on LLM failure
- `prompts/hypothesis.txt` - Externalized prompt template
- Documentation: `docs/features/hyde.md`
- E2E test scenarios for HyDE validation

#### Contextual Embeddings (`fitz_ai/ingestion/`)
- Summary-prefixed embedding: chunks are embedded as `f"{summary}\n\n{content}"` instead of just content
- Zero additional LLM calls - uses summaries already generated by enrichment pipeline
- Graceful fallback when no summary exists
- Implemented in both `IngestionPipeline` and `DiffIngestExecutor`
- Documentation: `docs/features/contextual-embeddings.md`

#### Query Rewriting (`fitz_ai/retrieval/rewriter/`)
- `QueryRewriter` - LLM-powered query transformation with conversation context
- `RewriteResult` - Structured result with rewritten query, confidence, and reasoning
- `ConversationContext` - Typed conversation history for pronoun resolution
- Rewrite types: conversational (pronouns), clarity (typos), retrieval (optimization)
- Ambiguity detection with multi-query expansion
- Single fast-tier LLM call per query (~100-200ms overhead)
- Graceful degradation on LLM failure (uses original query)
- `prompts/rewrite.txt` - Externalized prompt template
- Documentation: `docs/features/query-rewriting.md`
- Comprehensive test suite: `tests/unit/test_rewriter.py` (469 lines)

#### Conversational Context for SDK & API
- `fitz_ai/sdk/fitz.py` - `query()` now accepts `conversation_history` parameter
- `fitz_ai/api/routes/query.py` - POST `/query` accepts `conversation_history` in request body
- `fitz_ai/api/models/schemas.py` - Added conversation history to query schema
- Enables context-aware retrieval in programmatic use cases

#### Small Chunk Enrichment Skipper
- Skip LLM enrichment for chunks below minimum token threshold
- Configurable threshold in enrichment config
- Saves costs on small/trivial chunks

### 🔄 Changed

- `VectorSearchStep` now integrates query rewriter for conversational context resolution
- `FitzRagEngine.answer()` accepts optional conversation history
- Chat command passes conversation history to retrieval pipeline

### 🗑️ Removed

#### Dead Code Cleanup
- `fitz_ai/ingestion/enrichment/cache.py` - Unused summary cache (replaced by content-hash in state)
- `fitz_ai/ingestion/enrichment/router.py` - Unused enrichment router
- `fitz_ai/ingestion/enrichment/base.py` - Unused base class
- `fitz_ai/ingestion/enrichment/python_context.py` - Unused Python-specific context builder

### 🐛 Fixed

- Security tests now use proper mock fixtures
- Load/scalability tests fixed for CI stability
- Performance tests fixed with proper conftest setup
- Input validation tests corrected
- Rewriter prompt formatting fixes
- HyDE strategy integration fixes
- Various test fixture and configuration fixes

---

## [0.6.0] - 2026-01-21

### 🎉 Highlights

**Structured Data Module** - Complete rewrite of structured/tabular data handling with SQL generation, derived fields, schema detection, and unified vector+structured query routing. Enables natural language queries over CSV/tables with automatic SQL generation and result formatting.

**Fitz Cloud Integration** - Full integration with Fitz Cloud for query-time RAG optimization. Supports encrypted cache lookup/storage, model routing, and retrieval fingerprinting for cache keys.

**VLM Figure Description** - Docling parser now supports Vision Language Model (VLM) integration for automatic figure/chart description. When configured with a vision provider, images detected in documents are described by the VLM instead of showing "[Figure]" placeholders.

**Direct Text Ingestion** - Ingest text directly from command line without files: `fitz ingest "Your text here"`. Auto-detects text vs file paths.

**Architecture Overhaul** - Major refactoring to eliminate anti-patterns: typed models replace `dict[str, Any]`, god classes split into focused modules, global state eliminated, and Protocol-based type hints throughout. Test consolidation following DHH principles.

### 🚀 Added

#### Structured Data Module (`fitz_ai/structured/`)
- `schema.py` - Schema detection and field type inference (458 lines)
- `sql_generator.py` - Natural language to SQL translation (415 lines)
- `executor.py` - Safe SQL execution with sandboxing (404 lines)
- `derived.py` - Derived field computation (ratios, aggregates) (438 lines)
- `router.py` - Intelligent query routing (vector vs structured) (239 lines)
- `formatter.py` - Result formatting for LLM consumption (199 lines)
- `ingestion.py` - CSV/table ingestion with schema extraction (305 lines)
- `types.py` - Type definitions and protocols (369 lines)
- `constants.py` - SQL templates and constants (64 lines)
- `fitz tables` CLI command for table management (525 lines)
- Structured E2E test suite (989 lines)
- Vector search integration for derived fields

#### Direct Text Ingestion (`fitz_ai/cli/commands/ingest_direct.py`)
- `ingest_direct_text()` - Ingest text strings directly
- `is_direct_text()` - Auto-detect text vs file path
- `fitz ingest "Your text here"` - CLI support
- Automatic doc_id generation

#### Fitz Cloud (`fitz_ai/cloud/`)
- `CloudClient` - HTTP client for Fitz Cloud API
- Query-time RAG optimizer integration
- Model routing from cloud configuration
- Encrypted cache lookup and storage in RAGPipeline
- `retrieval_fingerprint` for deterministic cache keys
- `X-API-Key` header authentication

#### VLM Figure Description (`fitz_ai/ingestion/parser/plugins/docling.py`)
- `_describe_image_with_vlm()` - Sends detected figures to VLM for description
- `generate_picture_images=True` option in Docling pipeline
- PIL image extraction via `item.get_image(doc)`
- VLM call statistics tracking (`vlm_calls`, `vlm_errors`)
- 300s timeout for VLM calls (model loading on first call)

#### Docling Grid-Based Table Extraction
- `_build_table_from_grid()` - Extracts clean markdown tables from Docling's structured grid data
- Bypasses `export_to_markdown()` which adds unwanted bold formatting
- Proper column normalization and separator generation

#### Framework Integrations
- LangChain retriever abstraction layer

#### Figure E2E Tests (`tests/e2e/`)
- `FIGURE_RETRIEVAL` feature type in scenarios
- `figure_test.pdf` fixture with embedded bar chart
- 4 new scenarios (E145-E148) for figure content retrieval

### 🔄 Changed

#### Architecture Refactoring
- **Typed Models** - Replaced `dict[str, Any]` anti-pattern with proper dataclasses and typed models throughout codebase
- **Split `ingest.py`** (1,005 lines) into focused modules under `cli/commands/ingest/`
- **Split `init.py`** (1,033 lines) into focused modules under `cli/commands/init/`
- **Split `FitzPaths`** god class - Eliminated global state mutations
- **Protocol Type Hints** - Added Protocol-based type hints for documentation
- **`PluginKwargs`** - Typed class replacing `**kwargs` anti-pattern
- **Exception Handling** - Consolidated repeated exception handling in RAGPipeline
- **ClaraEngine** - Eliminated global state pollution

#### CLI Services Extraction
- `cli/services/ingest_service.py` - Extracted ingestion orchestration logic (319 lines)
- `cli/services/init_service.py` - Extracted initialization logic (275 lines)
- Clean separation of CLI presentation from business logic

#### Test Consolidation (DHH-style)
- Consolidated 12 granular RGS tests into `test_rgs_consolidated.py` (189 lines)
- Consolidated 6 context pipeline tests into `test_context_pipeline_consolidated.py` (106 lines)
- Removed ~300 lines of fragmented test files
- Each test file now tests a complete behavior, not implementation details

#### API Improvements
- Extracted API error decorator for consistent error handling
- Simplified tier resolution logic
- Added constants for magic values

#### Documentation
- `docs/api_reference.md` - New comprehensive API reference (233 lines)

#### Enrichment System
- E2E test rework for enrichment validation

### 🗑️ Removed

#### Consolidated Test Files (DHH-style cleanup)
- `test_context_pipeline_cross_file_dedupe.py`
- `test_context_pipeline_markdown_integrity.py`
- `test_context_pipeline_ordering.py`
- `test_context_pipeline_pack_boundary.py`
- `test_context_pipeline_unknown_group.py`
- `test_context_pipeline_weird_inputs.py`
- `test_rgs_chunk_id_fallback.py`
- `test_rgs_chunk_limit.py`
- `test_rgs_exclude_query.py`
- `test_rgs_max_chunks_limit.py`
- `test_rgs_metadata_format.py`
- `test_rgs_metadata_truncation.py`
- `test_rgs_no_citations.py`
- `test_rgs_prompt_core_logic.py`
- `test_rgs_prompt_slots.py`
- `test_rgs_strict_grounding_instruction.py`

### 🐛 Fixed

- **Table Markdown Formatting** - Tables no longer have bold headers (`**Column**`) that break downstream SQL generation
- **Security Test Assertion** - Fixed flaky security test assertion
- **Retrieval Latency Threshold** - Increased threshold for CI variance tolerance

### 📦 Configuration

#### VLM Configuration (`fitz_ai/llm/vision/local_ollama.yaml`)
- Increased endpoint timeout from 180s to 300s for VLM model loading
- Recommended model: `minicpm-v` for 16GB VRAM GPUs

---

## [0.5.2] - 2026-01-13

### 🎉 Highlights

**Multi-Hop Reasoning** - Fitz RAG now supports iterative multi-hop retrieval for complex queries requiring information synthesis across multiple documents. The system automatically detects when additional context is needed and performs follow-up retrievals.

**Entity Graph Expansion** - New entity graph system enriches retrieval by linking related chunks through shared entities. When a chunk mentions entities, the system automatically retrieves other chunks discussing the same concepts.

**Advanced Retrieval Intelligence** - Comprehensive suite of retrieval features now baked into the system including temporal queries, query expansion, hybrid search (dense+sparse), freshness/authority boosting, and aggregation query detection.

**End-to-End Testing Framework** - New comprehensive E2E test framework validates retrieval quality across diverse scenarios with automated validation and detailed reporting.

### 🚀 Added

#### Multi-Hop Reasoning (`fitz_ai/engines/fitz_rag/retrieval/multihop/`)
- `MultiHopController` - Orchestrates iterative retrieval with termination logic
- `InfoExtractor` - Extracts key information from intermediate results
- `CompletionEvaluator` - Determines when sufficient information has been gathered
- Configurable max hops and answer quality thresholds
- Multi-hop config in `FitzRagConfig` schema

#### Entity Graph System (`fitz_ai/ingestion/entity_graph/`)
- `EntityGraphStore` - Persistent storage for entity relationships
- Entity linking during ingestion enrichment
- Graph expansion step in retrieval pipeline
- Automatic retrieval of chunks sharing entities with top results
- Graph stored per collection in `.fitz/graphs/`

#### Retrieval Intelligence Suite (`fitz_ai/retrieval/`)
- **Temporal Queries** (`temporal/detector.py`) - Detects time-based comparisons and period filters
- **Query Expansion** (`expansion/expander.py`) - Generates synonym/acronym variations
- **Hybrid Search** (`sparse/index.py`) - BM25 sparse index with RRF fusion
- **Freshness & Authority** (`fitz_rag/retrieval/steps/freshness.py`) - Recency and authority boosting
- **Aggregation Queries** (`aggregation/detector.py`) - Detects statistical aggregation intent
- **Vocabulary System** (`vocabulary/`) - Exact keyword matching across chunks
  - `VocabularyDetector` - Extracts identifiers from content
  - `VocabularyMatcher` - Matches query terms to vocabulary
  - `VocabularyStore` - Persists keywords per collection
  - `VariationGenerator` - Generates term variations

#### End-to-End Testing (`tests/e2e/`)
- `E2ETestRunner` - Orchestrates full retrieval scenarios
- `TestReporter` - Generates detailed test reports
- `ScenarioValidator` - Validates retrieval results
- 15+ test scenarios covering:
  - Temporal queries (comparison, period filtering)
  - Sparse retrieval (exact keyword matching)
  - Tabular data routing
  - Conflict detection
  - Causal attribution
  - Code-aware search
  - Entity matching
- Test fixtures with structured markdown, code, CSV data

### 🔄 Changed

- **Module Organization**: Moved `vocabulary` and `entity_graph` modules from `fitz_ai/ingestion/` to `fitz_ai/retrieval/` for clearer separation of concerns
- **VectorSearchStep**: Now includes temporal handling, query expansion, hybrid search, multi-query expansion, and aggregation detection
- **EnrichmentPipeline**: Integrated entity graph construction during ingestion
- **Collection Delete**: Now cleans up entity graphs and vocabulary stores
- **README**: Major refactor with dedicated feature documentation pages in `docs/features/`
  - `aggregation-queries.md` - Statistical query handling
  - `code-aware-chunking.md` - Programming language support
  - `comparison-queries.md` - Entity comparison queries
  - `epistemic-honesty.md` - Constraint system
  - `freshness-authority.md` - Temporal relevance
  - `hierarchical-rag.md` - Multi-level summaries
  - `hybrid-search.md` - Dense + sparse fusion
  - `keyword-vocabulary.md` - Exact term matching
  - `multi-hop-reasoning.md` - Iterative retrieval
  - `multi-query-rag.md` - Query expansion
  - `query-expansion.md` - Synonym generation
  - `tabular-data-routing.md` - CSV/table handling
  - `temporal-queries.md` - Time-based filtering

### 🗑️ Removed

- `tools/smoketest/` - Replaced by E2E test framework
  - `smoke_fitz_rag_e2e.py` (750 lines)
  - `smoke_local_llm.py` (219 lines)

### 🐛 Fixed

- Tabular query handling now properly routes to registered tables
- Filesystem source plugin handles metadata more robustly
- Simple chunker improves overlap handling

---

## [0.5.1] - 2026-01-11

### 🎉 Highlights

**ChunkEnricher - Unified Enrichment Bus** - All chunk-level enrichment (summary, keywords, entities) is now baked in and runs automatically via a unified enrichment bus. The `ChunkEnricher` batches ~15 chunks per LLM call, making enrichment nearly free (~$0.13-0.74 for 1000 chunks).

**Exact Keyword Matching** - Keywords extracted during ingestion (test case IDs, ticket numbers, code identifiers) are now used for exact-match filtering at query time. Queries mentioning "TC-1001" will only return chunks containing that exact identifier.

**Multi-Query RAG** - Long or complex queries are automatically expanded into multiple focused search queries. 

**Comparison queries** - ("X vs Y") are detected and expanded to ensure both entities are retrieved.

**Table Registry** - CSV/table files are now reliably retrieved via a table registry that stores chunk IDs at ingestion time. No more missed tables due to low semantic similarity.

### 🚀 Added

#### ChunkEnricher (`fitz_ai/ingestion/enrichment/chunk/`)
- `ChunkEnricher` - Unified enrichment bus with extensible module architecture
- `EnrichmentModule` - Abstract base class for pluggable enrichment types
- `SummaryModule` - Generates searchable per-chunk summaries
- `KeywordModule` - Extracts exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- `EntityModule` - Extracts named entities (classes, people, technologies)
- Batched processing (~15 chunks per LLM call) for cost efficiency
- Keywords automatically saved to `VocabularyStore` for exact-match retrieval

#### Keyword Matching (`fitz_ai/engines/fitz_rag/retrieval/`)
- `KeywordMatcher` - Matches query terms against ingested vocabulary
- `VocabularyStore` - Persists auto-detected keywords per collection
- Keyword filtering in `VectorSearchStep` - filters results to chunks containing matched keywords

#### Multi-Query Expansion (`fitz_ai/engines/fitz_rag/retrieval/steps/vector_search.py`)
- Automatic query expansion for queries > 300 characters
- Comparison query detection (vs, compare, difference between)
- Comparison-aware expansion ensures both compared entities are retrieved
- Deduplication across expanded queries

#### Table Registry (`fitz_ai/tabular/registry.py`)
- `add_table_id()` / `get_table_ids()` - Store and retrieve table chunk IDs per collection
- Table IDs registered at ingestion time for reliable retrieval
- `retrieve()` method added to `VectorClient` protocol
- Table registry cleaned up on collection delete

### 🔄 Changed

- **Enrichment is now baked in**: Summary, keyword, and entity extraction run automatically when chat client is available
- **Removed opt-in config flags**: `enrichment.summary.enabled` and `enrichment.entities.enabled` removed
- **EnrichmentPipeline**: Now uses `ChunkEnricher` instead of separate summarizer and entity extractor
- **VectorClient protocol**: Added `retrieve(collection, ids)` method for direct ID-based lookup
- **Ingest UX**: Type a name to create new collection (no more "[0] + Create new" step)
- **Ingest UX**: Removed verbose "(docs corpus, hierarchical summaries)" and "VLM enabled" text
- **Documentation updated**: ENRICHMENT.md, INGESTION.md, CONFIG.md, ARCHITECTURE.md, README.md

### 🗑️ Removed

- `SummaryConfig` - No longer needed (summaries always on)
- `EntityConfig` - No longer needed (entities always on)
- `summaries_enabled` property on EnrichmentPipeline (replaced by `chunk_enrichment_enabled`)
- `entities_enabled` property on EnrichmentPipeline (replaced by `chunk_enrichment_enabled`)
- Dead code: `enabled_features` list in ingest command (was built but never displayed)

---

## [0.5.0] - 2026-01-07

### 🎉 Highlights

**Plugin Generator** - New `fitz plugin` command generates complete plugin scaffolds with templates, validation, and library context awareness. Generate chat, embedding, rerank, vision, chunker, retrieval, or constraint plugins with a single command.

**Parser Plugin System** - New parser abstraction replaces the reader module. Parsers handle document-to-structured-content conversion with plugins for plaintext, Docling (PDF/DOCX), and Docling+VLM (with figure description).

**Vision Plugin System** - Full YAML-based vision plugin support for VLM-powered figure description during ingestion. Supports Cohere, OpenAI, Anthropic, and Ollama vision models.

**Comprehensive Documentation** - Added 9 new documentation files covering API, architecture, configuration, constraints, enrichment, feature control, ingestion, SDK, and troubleshooting.

### 🚀 Added

#### Plugin Generator (`fitz_ai/plugin_gen/`)
- `fitz plugin generate` - Interactive plugin scaffolding wizard
- Template-based generation for all plugin types
- Library context awareness (detects installed packages)
- Validation and review workflow
- Templates for: `chunker`, `constraint`, `llm_chat`, `llm_embedding`, `llm_rerank`, `reader`, `retrieval`, `vector_db`

#### Parser Plugin System (`fitz_ai/ingestion/parser/`)
- `ParserRouter` - Routes files to appropriate parsers by extension
- `Parser` protocol with `can_parse()` and `parse()` methods
- `PlainTextParser` - Handles .txt, .md, .py, .json, etc.
- `DoclingParser` - PDF, DOCX, images via Docling library
- `DoclingVisionParser` - Docling + VLM for figure description
- `ParsedDocument` with typed `DocumentElement` structure

#### Vision Plugin System (`fitz_ai/llm/vision/`)
- YAML-based vision plugins matching chat/embedding pattern
- `cohere.yaml` - Cohere vision (command-a-vision-07-2025)
- `openai.yaml` - OpenAI vision (gpt-4o)
- `anthropic.yaml` - Anthropic vision (claude-sonnet-4)
- `local_ollama.yaml` - Ollama vision (llama3.2-vision)
- Vision plugin schema (`vision_plugin_schema.yaml`)
- Message transforms for vision requests

#### Source Abstraction (`fitz_ai/ingestion/source/`)
- `Source` protocol for file discovery
- `SourceFile` dataclass with URI, local path, metadata
- `FileSystemSource` plugin for local files

#### Documentation (`docs/`)
- `API.md` - REST API reference
- `ARCHITECTURE.md` - System design and layer dependencies
- `CONFIG.md` - Configuration reference
- `CONSTRAINTS.md` - Epistemic guardrails guide
- `ENRICHMENT.md` - Enrichment pipeline documentation
- `FEATURE_CONTROL.md` - Plugin-based feature control
- `INGESTION.md` - Ingestion pipeline guide
- `SDK.md` - Python SDK reference
- `TROUBLESHOOTING.md` - Common issues and solutions

#### CLI Improvements
- `fitz plugin` - New command for plugin generation
- `fitz init` - Vision model selection prompt added
- Vision provider/model configuration in init wizard

### 🔄 Changed

- **Parser replaces Reader**: `fitz_ai/ingestion/reader/` removed, replaced by `fitz_ai/ingestion/parser/`
- **Config schema**: `ExtensionChunkerConfig` now includes `parser` field for VLM control
- **Chunking router**: Now accepts parser selection via config
- **Init wizard**: Reordered sections (Chat → Embedding → Rerank → Vision → VectorDB)

### 🐛 Fixed

- `ParserRouter` no longer accepts invalid `vision_client` parameter
- Vision model defaults now use correct models (e.g., `command-a-vision-07-2025` not text model)
- Config validation accepts `parser` field in chunking config

### 🗑️ Removed

- `fitz_ai/ingestion/reader/` module (replaced by parser system)
- `fitz_ai/ingestion/chunking/engine.py` (consolidated into router)

---

## [0.4.5] - 2026-01-04

### 🎉 Highlights

**Zero-Friction Quickstart** - The `fitz quickstart` command now truly delivers on "zero-config RAG." Provider detection is fully automatic: Ollama detected → used; API key in environment → used; first time → guided through free Cohere signup. After initial setup, subsequent runs are completely prompt-free.

**CLIContext** - New centralized CLI context system provides a single source of truth for all configuration. Package defaults guarantee all values exist—no more scattered `.get()` fallbacks across commands.

**Collection Warnings** - The CLI now warns when a collection doesn't exist or is empty before querying, preventing confusing "I don't know" answers when the real issue is missing data.

### 🚀 Added

#### Zero-Friction Quickstart (`fitz_ai/cli/commands/quickstart.py`)
- **Auto-detection cascade**: Ollama → COHERE_API_KEY → OPENAI_API_KEY → guided signup
- `_resolve_provider()` - Detects best available LLM provider automatically
- `_check_ollama()` - Detects running Ollama with required models (llama3.2, nomic-embed-text)
- `_guide_cohere_signup()` - Step-by-step onboarding for new users (free tier)
- `_save_api_key_to_env()` - Cross-platform API key persistence (Windows: `.fitz/.env`, Unix: `.bashrc`/`.zshrc`)
- Removed engine selection prompt—quickstart now focuses on fitz_rag for simplicity

#### CLIContext (`fitz_ai/cli/context.py`)
- Centralized context for all CLI commands
- Guaranteed configuration values (package defaults always loaded)
- `get_collections()`, `require_collections()` - Collection discovery
- `select_collection()`, `select_engine()` - Interactive selection with validation
- `get_vector_db_client()`, `require_vector_db_client()` - Vector DB access
- `require_typed_config()` - Typed config with error handling
- `info_line()` - Single-line status display for commands

#### Config Loader (`fitz_ai/config/loader.py`)
- `load_engine_config()` - Loads merged config (package defaults + user overrides)
- `get_config_source()` - Returns config source for debugging
- Package defaults in `fitz_ai/engines/<engine>/config/default.yaml`

#### Collection Existence Warnings (`fitz_ai/cli/commands/query.py`)
- `_warn_if_collection_missing()` - Checks collection before query
- Warns when no collections exist: "Run 'fitz ingest ./docs' first"
- Warns when specified collection not found with available alternatives
- Warns when collection is empty (0 documents)

#### Engine Command (`fitz_ai/cli/commands/engine.py`)
- `fitz engine` - View or set default engine
- `fitz engine --list` - List all available engines
- Interactive selection with card-based UI
- Persists default engine to `.fitz/config.yaml`

#### Instrumentation System (`fitz_ai/core/instrumentation.py`)
- `BenchmarkHook` protocol for plugin performance measurement
- `register_hook()` / `unregister_hook()` - Thread-safe hook management
- `instrument()` decorator for method-level timing
- `create_instrumented_proxy()` - Transparent proxy wrapper for plugins
- Zero overhead when no hooks registered
- Tracks: layer, plugin name, method, duration, errors

#### Enterprise Plugin Discovery (`fitz_ai/cli/cli.py`)
- Auto-discovers `fitz-ai-enterprise` package when installed
- Adds `fitz benchmark` command from enterprise module
- Clean separation: core features in `fitz-ai`, advanced features in enterprise

#### CLI Map Tool (`tools/cli_map/`)
- New tool for analyzing CLI command structure
- Generates visual maps of command hierarchy

### 🔄 Changed

- **Engine rename**: `classic_rag` → `fitz_rag` for clearer branding
- **Quickstart simplified**: Removed `--engine` flag, focuses on fitz_rag for true zero-friction
- **README updated**: Documents auto-detection cascade and first-time experience
- **CLI commands**: All commands now use CLIContext instead of direct config loading
- **Documentation consolidated**: Removed outdated docs (CLARA.md, MIGRATION.md, release notes)

### 🐛 Fixed

- Quickstart no longer prompts for provider when API key is in environment
- Query command now warns about missing collections instead of returning "I don't know"
- Windows API key saving works correctly (uses `.fitz/.env` instead of shell config)

---

## [0.4.4] - 2025-12-30

### 🎉 Highlights

**GraphRAG Engine** - Full implementation of Microsoft's GraphRAG paradigm. Extract entities and relationships, build knowledge graphs, detect communities, and use local/global/hybrid search for relationship-aware retrieval.

**CLaRa Engine Rework** - Major refactoring of the compressed RAG engine with improved architecture and configuration.

**CLI Modernization** - Complete restructure of CLI UI into modular components for better maintainability and user experience.

**Semantic Constraints** - Constraint plugins now use embedding-based semantic matching instead of regex patterns, enabling language-agnostic conflict and causality detection.

### 🚀 Added

#### GraphRAG Engine (`fitz_ai/engines/graphrag/`)
- `GraphRAGEngine` - Knowledge graph-based retrieval engine
- Entity and relationship extraction via LLM (`graph/extraction.py`)
- Knowledge graph storage with NetworkX backend (`graph/storage.py`)
- Community detection using Louvain algorithm (`graph/community.py`)
- Community summarization for high-level insights
- Local search - find specific entities and relationships (`search/local.py`)
- Global search - query across community summaries (`search/global_search.py`)
- Hybrid search - combine local and global approaches
- Persistent storage via JSON serialization
- `fitz_ai/engines/graphrag/config/schema.py` - Full configuration schema

#### Semantic Matching (`fitz_ai/core/guardrails/semantic.py`)
- `SemanticMatcher` class for embedding-based concept detection
- Language-agnostic causal query detection
- Semantic conflict detection across chunks
- Configurable similarity thresholds
- Works with any embedding provider

#### CLI UI Modules (`fitz_ai/cli/ui/`)
- `console.py` - Shared Rich console instance
- `display.py` - Answer and result display formatting
- `engine_selection.py` - Interactive engine selection UI
- `output.py` - Structured output formatting
- `progress.py` - Progress bars and status indicators
- `prompts.py` - User input prompts and confirmations

#### Other
- `fitz_ai/cli/utils.py` - Shared CLI utilities
- `examples/clara_demo.py` - CLaRa engine demonstration
- `tests/test_graphrag_engine.py` - Comprehensive GraphRAG tests

### 🔄 Changed

- **CLaRa engine**: Major refactoring of `fitz_ai/engines/clara/engine.py` with improved architecture
- **CLI commands**: Enhanced `chat`, `ingest`, `init`, `query`, `quickstart` with new UI modules
- **Constraint plugins**: Refactored to use `SemanticMatcher` instead of regex patterns
  - `CausalAttributionConstraint` - Now uses semantic causal evidence detection
  - `ConflictAwareConstraint` - Now uses semantic conflict detection
  - `InsufficientEvidenceConstraint` - Simplified implementation
- **Hierarchy enricher**: Now accepts optional `SemanticMatcher` for conflict detection
- **Config loaders**: Improved engine configuration loading

### 🐛 Fixed

- Contract map tool no longer shows `<unknown>` SyntaxWarnings (added filename to ast.parse)
- Excluded `clara_model_cache` from contract map scanning
- Qdrant tests updated for YAML-based plugin system

---

## [0.4.3] - 2025-12-29

### 🎉 Highlights

**REST API** - New `fitz serve` command launches a FastAPI server with endpoints for query, ingest, and collection management. Build integrations without touching Python.

**SDK Module** - New `fitz_ai.sdk` provides a simplified high-level API for programmatic use. Import `from fitz_ai import Fitz` for quick access.

**Package Rename** - `fitz_ai/ingest/` renamed to `fitz_ai/ingestion/` for clearer naming. Adds new `reader` module for document reading abstraction.

### 🚀 Added

#### REST API (`fitz_ai/api/`)
- `fitz serve` - Launch FastAPI server for HTTP access
- `POST /query` - Query the knowledge base
- `POST /ingest` - Ingest documents
- `GET /collections` - List collections
- `GET /health` - Health check endpoint
- Dependency injection via `fitz_ai/api/dependencies.py`
- Pydantic schemas in `fitz_ai/api/models/schemas.py`

#### SDK Module (`fitz_ai/sdk/`)
- `Fitz` class as unified entry point
- Re-exported from `fitz_ai` package root
- Simplified API for common operations

#### Reader Module (`fitz_ai/ingestion/reader/`)
- `ReaderEngine` for document loading
- Plugin-based reader system
- `local_fs` plugin for local file reading

### 🔄 Changed

- **Package rename**: `fitz_ai/ingest/` → `fitz_ai/ingestion/`
- **Chunk model**: Moved from `fitz_ai/engines/fitz_rag/models/chunk.py` to `fitz_ai/core/chunk.py` for shared use across engines
- **Core exports**: `Chunk` now exported from `fitz_ai.core`

---

## [0.4.2] - 2025-12-28

### 🎉 Highlights

**Knowledge Map** - New `fitz map` command generates an interactive HTML visualization of your knowledge base. View document clusters, explore relationships, and identify coverage gaps. [EXPERIMENTAL]

**Hierarchical RAG** - New enrichment mode that generates multi-level summaries from your content. Groups related chunks and creates hierarchical context for improved retrieval.

**Fast/Smart Model Tiers** - LLM plugins now support two model tiers: "smart" for user-facing queries (best quality) and "fast" for background tasks like enrichment (best speed).

### 🚀 Added

#### Knowledge Map Visualization (`fitz_ai/map/`)
- `fitz map` - Generates interactive HTML knowledge graph
- Automatic clustering of related content
- Gap detection to identify missing coverage
- 2D projection of embeddings for visualization
- State caching for faster regeneration
- `--similarity-threshold` to control edge density
- `--rebuild` to force fresh embedding fetch
- `--no-open` to skip browser launch

#### Hierarchical Enrichment (`fitz_ai/ingest/enrichment/hierarchy/`)
- **HierarchyEnricher**: Generates multi-level summaries from chunks
- **ChunkGrouper**: Groups chunks by source file or custom rules
- **ChunkMatcher**: Filters chunks by path patterns
- Simple mode (zero-config) with smart defaults
- Rules mode for power-users with custom configuration
- Centralized prompts in `fitz_ai/prompts/hierarchy/`

#### Content Type Detection (`fitz_ai/ingest/detection.py`)
- Auto-detects codebase vs document corpus
- Recognizes project markers (pyproject.toml, package.json, Cargo.toml, etc.)
- Selects appropriate enrichment strategy automatically

#### LLM Model Tiers
- `models.smart` and `models.fast` in YAML plugin defaults
- `tier="smart"` or `tier="fast"` parameter for client creation
- Smart defaults: `command-a-03-2025` (Cohere), `gpt-4o` (OpenAI)
- Fast defaults: `command-r7b-12-2024` (Cohere), `gpt-4o-mini` (OpenAI)

#### Comprehensive CLI Tests
- `test_cli_chat.py` - Chat command tests
- `test_cli_collections.py` - Collection management tests
- `test_cli_config.py` - Config command tests
- `test_cli_doctor.py` - System diagnostics tests
- `test_cli_ingest.py` - Ingestion pipeline tests
- `test_cli_init.py` - Initialization tests
- `test_cli_map.py` - Knowledge map tests
- `test_cli_query.py` - Query command tests
- `test_local_llm_*.py` - Local LLM runtime tests

### 🔄 Changed

- Chunker plugins reorganized: `simple.py` and `recursive.py` moved to `plugins/default/`
- `fitz ingest` now supports `-H/--hierarchy` flag for hierarchical enrichment
- Contract map tool refactored with improved autodiscovery
- YAML plugin `defaults.model` replaced with `defaults.models.{smart,fast}` structure

### 🐛 Fixed

- Various fixes to contract map analysis
- Improved chunking router registry handling

---

## [0.4.1] - 2025-12-27

### 🐛 Fixed

- Minor fixes and improvements

---

## [0.4.0] - 2025-12-26

### 🎉 Highlights

**Conversational RAG** - New `fitz chat` command for interactive multi-turn conversations with your knowledge base. Each turn retrieves fresh context while maintaining conversation history.

**Enrichment Pipeline** - New semantic enrichment system that enhances chunks with LLM-generated summaries and produces project-level artifacts for improved retrieval context.

**Batch Embedding** - Automatic batch size adjustment with recursive halving on failure. Significantly faster ingestion for large document sets.

**Collection Management CLI** - New `fitz collections` command for interactive vector DB management.

### 🚀 Added

#### Enrichment System (`fitz_ai/ingest/enrichment/`)
- **EnrichmentPipeline**: Unified entry point for all enrichment operations
- **ChunkSummarizer**: LLM-generated descriptions for each chunk to improve search
- **Artifact Generation**: Project-level insights stored and retrieved with queries
  - `architecture_narrative` - High-level codebase description
  - `data_model_reference` - Data structures and models
  - `dependency_summary` - External dependency overview
  - `interface_catalog` - Public APIs and interfaces
  - `navigation_index` - Codebase navigation guide
- **Context Plugins**: File-type specific context builders (Python, generic)
- **SummaryCache**: Hash-based caching to avoid re-summarizing unchanged content
- **EnrichmentRouter**: Routes documents to appropriate enrichers by file type

#### Batch Embedding
- `embed_batch()` method on `EmbeddingClient`
- Automatic batch size adjustment (starts at 96)
- Recursive halving on API failures
- Progress logging per batch

#### Conversational Interface
- `fitz chat` - Interactive conversation with your knowledge base
- `-c, --collection` option to specify collection directly
- Collection selection on startup (prompts if not specified)
- Per-turn retrieval with conversation history (last 15 messages)
- Rich UI with styled panels for user/assistant messages
- `display_sources()` utility for consistent source table display (vector score, rerank score, excerpt)
- Graceful exit handling (Ctrl+C, 'exit', 'quit')

#### Documentation
- Expanded CLI documentation in `docs/CLI.md` with chat command examples

#### CLI Improvements
- `fitz collections` - Interactive collection management
- Enhanced `fitz_ai/cli/ui.py` with Rich console utilities
- Improved ingest command with enrichment support

#### Retrieval Pipeline
- `ArtifactFetchStep` - Prepends artifacts to every query result (score=1.0)
- Artifacts provide consistent codebase context for all queries

### 🔄 Changed

- Ingest executor now integrates enrichment pipeline
- Ingestion state schema includes enrichment metadata
- README simplified and updated

---

## [0.3.6] - 2025-12-23

### 🎉 Highlights

**Quickstart Command** - Zero-friction entry point for new users. Get a working RAG system in ~5 minutes with just `pip install fitz-ai` and `fitz quickstart`.

**Incremental Ingestion** - Content-hash-based incremental ingestion that skips unchanged files. State-file-authoritative architecture enables user-implemented vector DB plugins without requiring scroll/filter APIs.

**File-Type Based Chunking** - Intelligent routing to specialized chunkers based on file extension. Markdown, Python, and PDF each get purpose-built chunking strategies.

**Epistemic Safety Layer** - Constraint plugins and answer modes prevent overconfident answers when evidence is insufficient, disputed, or lacks causal attribution.

**YAML Retrieval Pipelines** - Retrieval strategies now defined in YAML. Compose steps like `vector_search → rerank → threshold → limit` declaratively.

### 🚀 Added

#### Quickstart Experience
- `fitz quickstart` command for zero-config RAG setup
- Interactive mode with path/question prompts
- Direct mode: `fitz quickstart ./docs "question"`
- Auto-prompts for Cohere API key, offers to save to shell config
- Auto-generates `.fitz/config.yaml` on first run
- Uses Cohere + local FAISS (no external services required)

#### Incremental Ingestion System
- Content-hash-based file tracking in `.fitz/ingest.json`
- Files skipped if content hash matches previous ingestion
- `--force` flag to bypass skip logic and re-ingest everything
- `FileScanner`: Walks directories, filters by supported extensions
- `Differ`: Computes ingestion plan (new/changed/deleted files)
- `DiffIngestExecutor`: Orchestrates parse → chunk → embed → upsert
- `IngestStateManager`: Persists and queries ingestion state

#### File-Type Based Chunking
- `ChunkingRouter`: Routes documents to file-type specific chunkers
- Per-extension chunker configuration via `by_extension` map
- Config ID tracking (`chunker_id`, `parser_id`, `embedding_id`) for re-chunking detection
- `MarkdownChunker`: Splits on headers, preserves code blocks
- `PythonCodeChunker`: AST-based splitting by class/function, includes imports
- `PdfSectionChunker`: Detects ALL CAPS headers, numbered sections, keyword sections

#### Constraint Plugin System
- `ConflictAwareConstraint`: Detects contradicting classifications across chunks
- `InsufficientEvidenceConstraint`: Blocks confident answers when evidence is weak
- `CausalAttributionConstraint`: Prevents implicit causality synthesis
- `ConstraintResult` with `allow_decisive_answer`, `reason`, `signal` fields

#### Answer Mode System
- `AnswerMode` enum: `CONFIDENT`, `QUALIFIED`, `DISPUTED`, `ABSTAIN`
- `AnswerModeResolver`: Maps constraint signals to answer mode
- Mode-specific LLM instruction prefixes for epistemic tone control
- `mode` field added to `RGSAnswer` and core `Answer`

#### YAML Retrieval Pipelines
- `dense.yaml` and `dense_rerank.yaml` pipeline definitions
- Modular retrieval steps: `vector_search`, `rerank`, `threshold`, `limit`, `dedupe`
- `RetrievalPipelineFromYaml` with `retrieve()` method
- Step registry with `get_step_class()` and `list_available_steps()`

#### CLI Improvements
- `fitz init` prompts for chunking configuration
- `fitz ingest` loads chunking config from `fitz.yaml`
- `fitz query --retrieval/-r` flag for retrieval strategy selection
- Shared `display_answer()` for consistent output formatting

### 🔄 Changed

- Config field `retriever` → `retrieval` across codebase
- State schema requires `chunker_id`, `parser_id`, `embedding_id` fields
- `IngestStateManager.mark_active()` requires config ID parameters
- `DiffIngestExecutor` takes `chunking_router` instead of single chunker
- FAISS moved to base dependencies (not optional)

### 🗑️ Deprecated

- `OverlapChunker`: Use `SimpleChunker` with `chunk_overlap` instead

### 🐛 Fixed

- Threshold regression for temporal/causal queries (reordered pipeline steps)
- Plugin discovery paths for YAML-based plugins
- Windows path separator issue in scanner tests
- Contract map now correctly discovers all 25 plugins

---

## [0.3.5] - 2025-12-21

### 🎉 Highlights
**Plugin Schema Standardization** - All LLM plugin YAMLs now follow an identical structure with master schema files as the single source of truth. Adding new providers is now more predictable and self-documenting.

**Generic HTTP Vector DB Plugin System** - HTTP-based vector databases (Qdrant, Pinecone, Weaviate, Milvus) now work with just a YAML config drop - no Python code needed. The same plugin interface works for both HTTP and local vector DBs.

### 🚀 Added
- **Master schema files** for plugin validation and defaults
  - `fitz_ai/llm/schemas/chat_plugin_schema.yaml`
  - `fitz_ai/llm/schemas/embedding_plugin_schema.yaml`
  - `fitz_ai/llm/schemas/rerank_plugin_schema.yaml`
  - `fitz_ai/vector_db/schemas/vector_db_plugin_schema.yaml` - documents all YAML fields for vector DB plugins
- **Schema defaults loader** `fitz_ai/llm/schema_defaults.py` - reads defaults from YAML schemas instead of hardcoding in Python
- **FAISS admin methods** - `list_collections()`, `get_collection_stats()`, `scroll()` for feature parity with HTTP-based vector DBs
- **Azure OpenAI embedding plugin** `fitz_ai/llm/embedding/azure_openai.yaml`
- **New vector DB plugins** (YAML-only, no Python needed):
  - `fitz_ai/vector_db/plugins/pinecone.yaml` - Pinecone cloud vector DB
  - `fitz_ai/vector_db/plugins/weaviate.yaml` - Weaviate vector DB
  - `fitz_ai/vector_db/plugins/milvus.yaml` - Milvus vector DB
- **Vector DB base class for local plugins** `fitz_ai/vector_db/base_local.py` - reduces boilerplate when implementing local vector DBs
- **Comprehensive plugin tests** `tests/test_plugin_system.py` covering chat, embedding, rerank, and FAISS
- **Vector DB plugin tests** `tests/test_generic_vector_db_plugin.py` - validates YAML loading, HTTP operations, point transformation, UUID conversion, and auth handling

### 📄 Changed
- **Standardized plugin YAML structure** - All 13 LLM plugins now follow identical section ordering:
```
  IDENTITY → PROVIDER → AUTHENTICATION → REQUIRED_ENV → HEALTH_CHECK → ENDPOINT → DEFAULTS → REQUEST → RESPONSE
```
- **Chat plugins updated**: openai, cohere, anthropic, local_ollama, azure_openai
- **Embedding plugins updated**: openai, cohere, local_ollama, azure_openai
- **Rerank plugins updated**: cohere
- **Renamed** `list_yaml_plugins()` → `list_plugins()` (removed redundant "yaml" prefix)
- **Loader applies defaults** from master schema - missing optional fields get default values automatically
- **Updated `qdrant.yaml`** - added `count` and `create_collection` operations for full feature parity

### 🛠️ Improved
- **Single source of truth** - Field definitions, types, defaults, and allowed values all live in schema YAMLs
- **Self-documenting schemas** - Each field has `description` and `example` in the schema
- **Forward compatibility** - New fields with defaults don't break existing plugin YAMLs
- **Consistent vector DB interface** - FAISS now implements same admin methods as Qdrant, no backend-specific code needed
- **Generic HTTP vector DB loader** - `GenericVectorDBPlugin` executes YAML specs for any HTTP-based vector DB with support for:
  - All standard operations: `search`, `upsert`, `count`, `create_collection`, `delete_collection`, `list_collections`, `get_collection_stats`
  - Auto-collection creation on 404
  - Point format transformation (standard → provider-specific)
  - UUID conversion for DBs that require it (e.g., Qdrant)
  - Flexible auth (bearer, custom headers, optional)
  - Jinja2 templating for endpoints and request bodies
- **`available_vector_db_plugins()`** - lists all available plugins (both YAML and local)

### 🐛 Fixed
- **FAISS missing interface methods** - Added `list_collections()`, `get_collection_stats()`, `scroll()` to match vector DB contract
- **Rerank mock in tests** - Fixed `MockRerankEngine` to return `List[Tuple[int, float]]` instead of flat list

---

## [0.3.4] - 2025-12-19

### 🎉 Pypi-Release

**https://pypi.org/project/fitz-ai/**

---

## [0.3.3] - 2025-12-19

### 🎉 Highlights

**YAML-based Plugin System** - LLM and Vector DB plugins are now defined entirely in YAML, not Python. Adding new providers is now as simple as creating a YAML file.

### 🚀 Added

- **YAML-based LLM plugins**: Chat, Embedding, and Rerank plugins now use YAML specs
  - `fitz_ai/llm/chat/*.yaml` - Chat plugins (OpenAI, Cohere, Anthropic, Azure, Ollama)
  - `fitz_ai/llm/embedding/*.yaml` - Embedding plugins  
  - `fitz_ai/llm/rerank/*.yaml` - Rerank plugins
- **YAML-based Vector DB plugins**: Vector databases now use YAML specs
  - `fitz_ai/vector_db/plugins/qdrant.yaml`
  - `fitz_ai/vector_db/plugins/pinecone.yaml`
  - `fitz_ai/vector_db/plugins/local_faiss.yaml`
- **Generic plugin runtime**: `GenericVectorDBPlugin` and `YAMLPluginBase` execute YAML specs at runtime
- **Provider-agnostic features**: YAML `features` section for provider-specific behavior
  - `requires_uuid_ids`: Auto-convert string IDs to UUIDs
  - `auto_detect`: Service discovery configuration
- **Message transforms**: Pluggable message format transformers for different LLM APIs
  - `openai_chat`, `cohere_chat`, `anthropic_chat`, `ollama_chat`, `gemini_chat`

### 🔄 Changed

- **LLM plugins**: Migrated from Python classes to YAML specifications
- **Vector DB plugins**: Migrated from Python classes to YAML specifications  
- **Plugin discovery**: Now scans `*.yaml` files instead of `*.py` modules
- **fitz_ai/core/registry.py**: Single source of truth for all plugin access

### 🐛 Fixed

- **Qdrant 400 Bad Request**: String IDs now converted to UUIDs automatically
- **Auto-create collection**: Collections created on first upsert (handles 404)
- **Import errors in CLI**: Fixed by adding re-exports to `fitz_ai/core/registry.py`

---

## [0.3.2] - 2025-12-18

### 🔄 Changed

- Renamed config field `llm` → `chat` for clarity (breaking change - regenerate config with `fitz init`)

### 🚀 Added

- `fitz db` command to inspect vector database collections
- `fitz chunk` command to preview chunking strategies
- `fitz query` as top-level command (was `fitz pipeline query`)
- `fitz config` as top-level command (was `fitz pipeline config show`)
- LAN scanning for Qdrant detection in `fitz init`
- Auto-select single provider options in `fitz init`

### 🐛 Fixed

- Contract map now discovers re-exported plugins (local-faiss)
- Contract map health check false positives removed
- Test fixes for `llm` → `chat` rename

---

## [0.3.1] - 2025-01-17

### 🐛 Fixed

- **CLI Import Error**: Fixed misleading error messages when internal fitz modules fail to import
- **Detection Module**: Moved `fitz_ai/cli/detect.py` to `fitz_ai/core/detect.py` as single source of truth for service detection
- **FAISS Detection**: `SystemStatus.faiss` now returns `ServiceStatus` instead of boolean for consistent API
- **Registry Exceptions**: `LLMRegistryError` now inherits from `PluginNotFoundError` for consistent exception handling
- **Invalid Plugin Type**: `get_llm_plugin()` now raises `ValueError` for invalid plugin types (not just unknown plugins)
- **Ingest CLI**: Fixed import of non-existent `available_embedding_plugins` now uses `available_llm_plugins("embedding")`
- **UTF-8 Encoding**: Added encoding declaration to handle emoji characters in error messages on Windows

### 🔄 Changed

- `fitz_ai/core/detect.py` is now the canonical location for all service detection (Ollama, Qdrant, FAISS, API keys)
- `SystemStatus` now has `best_llm`, `best_embedding`, `best_vector_db` helper properties
- CLI modules (`doctor.py`, `init.py`) now import from `fitz_ai.core.detect` instead of `fitz_ai.cli.detect`

---

## [0.3.0] - 2024-12-17

### 🎉 Overview

Fitz v0.3.0 transforms the project from a RAG framework into a **multi-engine knowledge platform**. This release introduces a pluggable engine architecture, the CLaRa engine for compression-native RAG, and a universal runtime for seamless engine switching.

### ✨ Highlights

- **Universal Runtime**: `run(query, engine="clara")` switch engines with one parameter
- **Engine Registry**: Discover, register, and manage knowledge engines
- **Protocol-Based Design**: Implement `answer(Query) -> Answer` to create custom engines
- **CLaRa Engine**: Apple's Continuous Latent Reasoning with 16x-128x document compression

### 🚀 Added

#### Core Contracts (`fitz_ai/core/`)
- `KnowledgeEngine` protocol for paradigm-agnostic engine interface
- `Query` dataclass for standardized query representation with constraints
- `Answer` dataclass for standardized response with provenance
- `Provenance` dataclass for source attribution
- `Constraints` dataclass for query-time limits (max_sources, filters)
- Exception hierarchy: `QueryError`, `KnowledgeError`, `GenerationError`, `ConfigurationError`

#### Universal Runtime (`fitz_ai/runtime/`)
- `run(query, engine="...")` universal entry point
- `EngineRegistry` for global engine discovery and registration
- `create_engine(engine="...")` factory for engine instances
- `list_engines()` to discover available engines
- `list_engines_with_info()` for engines with descriptions

#### CLaRa Engine (`fitz_ai/engines/clara/`)
- `ClaraEngine` full implementation of CLaRa paradigm
- `run_clara()` convenience function for quick queries
- `create_clara_engine()` factory for reusable instances
- `ClaraConfig` comprehensive configuration
- Auto-registration with global engine registry
- 17 passing tests covering all functionality

#### Fitz RAG Engine (`fitz_ai/engines/fitz_rag/`)
- `FitzRagEngine` wrapper implementing `KnowledgeEngine`
- `run_fitz_rag()` convenience function
- `create_fitz_rag_engine()` factory function
- Auto-registration with global engine registry

### 🔄 Changed

#### Public API (BREAKING)
- Entry points: `RAGPipeline.from_config(config).run()` → `run_fitz_rag()`
- Answer format: `RGSAnswer.answer` → `Answer.text`
- Source format: `RGSAnswer.sources` → `Answer.provenance`
- Chunk ID: `source.chunk_id` → `provenance.source_id`
- Text excerpt: `source.text` → `provenance.excerpt`

#### Directory Structure
```
OLD (v0.2.x):
fitz_ai/
├── pipeline/          # RAG-specific
├── retrieval/         # RAG-specific
├── generation/        # RAG-specific
└── core/              # Mixed concerns

NEW (v0.3.0):
fitz_ai/
├── core/              # Paradigm-agnostic contracts
├── engines/
│   ├── fitz_rag/   # Traditional RAG
│   └── clara/         # CLaRa engine
├── runtime/           # Multi-engine orchestration
├── llm/               # Shared LLM service
├── vector_db/         # Shared vector DB service
└── ingest/            # Shared ingestion
```

### 🐛 Fixed

- Resolved all circular import dependencies
- Fixed import path inconsistencies across modules
- Corrected Provenance field usage (score → metadata)
- Fixed engine registration order to prevent import errors
- Proper lazy imports in runtime to avoid circular dependencies

### 📚 Documentation

- Updated README with multi-engine architecture
- Added CLaRa hardware requirements
- Migration guide for v0.2.x → v0.3.0
- Updated all code examples

### 🧪 Testing

- All existing tests updated and passing
- 17 new tests for CLaRa engine (config, engine, runtime, registration)
- Tests use mocked dependencies (no GPU required for testing)
- Integration tests for engine protocol compliance

### ⚠️ Breaking Changes

1. **Import paths changed**: Update all imports (see Migration Guide)
2. **Public API changed**: Use `run_fitz_rag()` or engine-specific functions
3. **Answer format changed**: `Answer.text` and `Answer.provenance`
4. **No backwards compatibility layer**: Clean break for cleaner codebase

### 📦 Dependencies

New optional dependencies:
```toml
[project.optional-dependencies]
clara = ["transformers>=4.35.0", "torch>=2.0.0"]
```

---

## [0.2.0] - 2025-12-16

### 🎉 Overview

Quality-focused release with enhanced observability, local-first development, and production readiness improvements.

### ✨ Highlights

- **Contract Map Tool**: Living architecture documentation with automatic quality tracking
- **Ollama Integration**: Use local LLMs (Llama, Mistral, etc.) with zero API costs
- **FAISS Support**: Local vector database for development and testing
- **Production Readiness**: 100% appropriate error handling, zero architecture violations

### 🚀 Added

#### Quality Tools
- Contract map with Any usage analysis
- Exception pattern detection
- Code quality metrics tracking
- Architecture violation detection

#### Local Runtime
- Ollama backend for chat, embedding, rerank
- FAISS local vector database
- Local development workflow

#### Developer Experience
- Enhanced error messages in API clients
- Improved logging for file operations
- Better type hints throughout
- Comprehensive documentation

### 🔄 Changed

- Error handling with comprehensive logging
- Type safety improved (92% clean)
- API error messages with better context

### 📚 Documentation

- Updated README with v0.2.0 features
- Contract Map tool documentation
- Local development guide

---

## [0.1.0] - 2025-12-14

### 🎉 Overview

Initial release of Fitz RAG framework.

### 🚀 Added

- Core RAG pipeline
- OpenAI, Azure, Cohere LLM plugins
- Qdrant vector database integration
- Document ingestion pipeline
- CLI tools for query and ingestion

---

[Unreleased]: https://github.com/yafitzdev/fitz-ai/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/yafitzdev/fitz-ai/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/yafitzdev/fitz-ai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/yafitzdev/fitz-ai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/yafitzdev/fitz-ai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.5...v0.5.0
[0.4.5]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/yafitzdev/fitz-ai/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.6...v0.4.0
[0.3.6]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/yafitzdev/fitz-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yafitzdev/fitz-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz-ai/releases/tag/v0.1.0
