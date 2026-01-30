# Technology Stack

**Analysis Date:** 2026-01-30

## Languages

**Primary:**
- Python 3.10+ - Core codebase, RAG engine, CLI, API

**Secondary:**
- YAML - Configuration and plugin definitions
- SQL - PostgreSQL queries via psycopg
- JSON - Data serialization and API responses

## Runtime

**Environment:**
- Python 3.10, 3.11, 3.12
- Supports Windows, macOS, Linux (cross-platform)

**Package Manager:**
- pip (with setuptools)
- Lockfile: Not detected (relies on pinned versions in `pyproject.toml`)

## Frameworks

**Core:**
- `pydantic>=2.0` - Data validation (core models, API schemas)
- `typing-extensions>=4.7` - Type hint compatibility
- `jinja2>=3.1` - Template rendering for prompts

**CLI:**
- `typer>=0.9` - CLI framework for 14 commands (fitz init, ingest, query, etc.)

**API:**
- `fastapi>=0.100.0` - REST API framework (optional dependency, extra: `api`)
- `uvicorn[standard]>=0.23.0` - ASGI server (optional dependency, extra: `api`)

**Configuration & Serialization:**
- `pyyaml>=6.0` - YAML config parsing
- `httpx>=0.24` - Async HTTP client for external services

**Cryptography:**
- `cryptography>=41.0` - AES-256-GCM encryption for Fitz Cloud cache

**Testing:**
- `pytest>=7.0` - Test runner
- `pytest-cov>=4.0` - Coverage reporting
- `pytest-xdist>=3.0` - Parallel test execution
- `hypothesis>=6.100.0` - Property-based testing (dev extra)
- `mutmut>=2.4,<3` - Mutation testing (dev extra)

**Code Quality:**
- `black>=23.0` - Code formatter (dev extra)
- `isort>=5.0` - Import sorting (dev extra)
- `mypy>=1.0` - Type checking (dev extra)
- `ruff>=0.1.0` - Linter (dev extra)

**Performance Testing:**
- `locust>=2.20` - Load testing (optional, extra: `loadtest`)
- `psutil>=5.9` - System metrics (dev and loadtest extras)

## Key Dependencies

**Critical Storage:**
- `psycopg[binary]>=3.1` - PostgreSQL client (binary builds included)
- `psycopg-pool>=3.1` - Connection pooling for PostgreSQL
- `pgvector>=0.2.0` - PostgreSQL pgvector extension client (vector operations)
- `fitz-pgserver>=0.1.5` - Fork of pgserver with Windows crash recovery fix (embedded PostgreSQL for local mode)

**Document Processing:**
- `docling>=2.0` - Advanced document parsing (PDF, DOCX, PPTX, images, HTML)
  - Includes layout analysis, table extraction, OCR, figure detection
  - Optional VLM integration for image descriptions

**Vector Databases (Optional):**
- `qdrant-client>=1.7` - Qdrant vector DB client (optional, extra: `remote`)
- `faiss-cpu>=1.7.0` - FAISS vector DB (legacy support, optional, extra: `faiss`)

**Local LLM:**
- `ollama>=0.1.0` - Ollama LLM provider for local models (optional, extra: `local`)

**Advanced ML (CLaRa Engine):**
- `torch>=2.0` - PyTorch (optional, extra: `clara`)
- `transformers>=4.35` - Hugging Face transformers (optional, extra: `clara`)
- `accelerate>=0.24` - Training acceleration (optional, extra: `clara`)
- `bitsandbytes>=0.41` - Quantization (optional, extra: `clara`)
- `peft>=0.10` - Parameter-efficient fine-tuning (optional, extra: `clara`)

**Framework Integrations (Optional):**
- `langchain-core>=0.2.0` - LangChain integration (optional, extra: `langchain`)
- `llama-index-core>=0.10.0` - LlamaIndex integration (optional, extra: `llamaindex`)

**Visualization (Optional):**
- `umap-learn>=0.5.0` - Knowledge map visualization (optional, extra: `map`)
- `scikit-learn>=1.0` - ML utilities (optional, extra: `map`)

## Configuration

**Environment:**
- Config-driven via YAML files in `~/.fitz/config/`
- Provider selection through string specs: `"provider"` or `"provider/model"`
  - Example: `chat: "anthropic/claude-sonnet-4"` or `chat: "cohere"`
- Layered loading:
  1. Package defaults (e.g., `fitz_ai/engines/fitz_rag/config/default.yaml`)
  2. User overrides (e.g., `~/.fitz/config/fitz_rag.yaml`)
  3. Deep merge (every config value always exists)

**Key Configuration Sections:**
- `chat` - Chat provider (required)
- `embedding` - Embedding provider (required)
- `rerank` - Reranker (optional, null to disable)
- `vision` - Vision/VLM provider (optional, null to disable)
- `vector_db` - Vector DB selection (`pgvector` default)
- `vector_db_kwargs` - Connection: mode (`local` or `external`), data_dir, connection_string, HNSW settings
- `retrieval_plugin` - Retrieval strategy (`dense` or `dense_rerank`)
- `collection` - Vector DB collection name (required)
- `chunking` - Parser choice, chunk size, overlap
- `cloud` - Fitz Cloud cache API (optional)

**Build:**
- `pyproject.toml` - Package metadata, dependencies, build configuration
- `.github/workflows/` - CI/CD pipeline (GitHub Actions)
  - `ci.yml` - Linting, unit/integration/E2E tests
  - `mutation.yml` - Mutation testing
  - `release.yml` - Release automation

**Environment Files:**
- `.fitz/config/fitz_rag.yaml` - User configuration for RAG engine
- `tests/integration/.env.integration.example` - Template for cloud integration test secrets

## Platform Requirements

**Development:**
- Python 3.10+ with venv
- PostgreSQL 14+ (or embedded via fitz-pgserver)
- ~4GB RAM minimum (depends on vector DB size)
- Windows/macOS/Linux support

**Production:**
- Python 3.10+
- PostgreSQL 14+ with pgvector extension (or embedded PostgreSQL via fitz-pgserver)
- Optional: External vector DB (Qdrant, etc.)
- Optional: Fitz Cloud API for cache/routing

## Notable Architectural Choices

**Plugin System:**
- YAML-based plugins for external services (Chat, Embedding, Rerank, Vision providers)
- Python-based plugins for internal logic (Parser, Chunking, Guardrails)
- Unified registry pattern for discovery and initialization

**Storage Architecture:**
- PostgreSQL + pgvector as primary storage (vectors + metadata + tables)
- Support for both local (embedded fitz-pgserver) and external PostgreSQL
- HNSW indexing for vector search (99% recall, zero maintenance)

**No ORM:**
- Direct psycopg SQL queries (explicit, performant)
- Connection pooling via psycopg_pool

**Encryption:**
- AES-256-GCM for Fitz Cloud cache (org_key NEVER sent to server)
- Local-first design: encryption happens client-side

**Cross-Platform:**
- Windows symlink fix for Hugging Face model caching (Docling dependency)
- Signal handling for graceful shutdown (SIGTERM, SIGINT)

---

*Stack analysis: 2026-01-30*
