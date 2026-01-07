# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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

[Unreleased]: https://github.com/yafitzdev/fitz-ai/compare/v0.5.0...HEAD
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
