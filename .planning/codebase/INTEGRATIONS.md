# External Integrations

**Analysis Date:** 2026-01-30

## APIs & External Services

**LLM Chat Providers:**
- **Anthropic (Claude)** - Chat provider
  - SDK: `anthropic` (lazy-imported in `fitz_ai/llm/providers/anthropic.py`)
  - Auth: Bearer token from environment variable `ANTHROPIC_API_KEY`
  - Models: claude-opus, claude-sonnet-4, claude-haiku (tier-based: smart/balanced/fast)
  - Usage: Generate answers from retrieved context

- **OpenAI (GPT-4o)** - Chat provider
  - SDK: `openai` (lazy-imported in `fitz_ai/llm/providers/openai.py`)
  - Auth: Bearer token from environment variable `OPENAI_API_KEY`
  - Models: gpt-4o (smart), gpt-4o-mini (balanced/fast)
  - Supports: Azure OpenAI via custom base_url configuration
  - Usage: Generate answers, vision capabilities

- **Cohere** - Chat provider
  - SDK: `cohere` (lazy-imported in `fitz_ai/llm/providers/cohere.py`)
  - Auth: X-Api-Key header from environment variable
  - Models: command-a-03-2025 (smart), command-r7b-12-2024 (balanced/fast)
  - Default in `fitz_ai/engines/fitz_rag/config/default.yaml`
  - Usage: Chat, embedding, reranking (unified provider)

- **Ollama (Local)** - Local LLM provider
  - SDK: `ollama` (lazy-imported in `fitz_ai/llm/providers/ollama.py`)
  - Connection: HTTP client to local Ollama instance (default: http://localhost:11434)
  - Models: Any model available in local Ollama (e.g., nomic-embed-text, qwen2.5)
  - Optional dependency: Install with `pip install "fitz-ai[local]"`
  - Usage: Local-only chat and embeddings (no internet required)

**Embedding Providers:**
- **OpenAI** - text-embedding-3-small (1536-dim)
  - SDK: `openai`
  - Auth: Bearer token

- **Cohere** - embed-multilingual-v3.0 (1024-dim)
  - SDK: `cohere`
  - Auth: X-Api-Key

- **Voyage AI** - voyage-2
  - SDK: Lazy-imported via httpx (no official Python SDK)
  - Auth: Bearer token

- **Ollama** - Local embeddings (e.g., nomic-embed-text, 768-dim)
  - SDK: `ollama`
  - Connection: Local HTTP

**Reranking Providers:**
- **Cohere Rerank** - rerank-multilingual-v3.0
  - SDK: `cohere` (CohereRerank class)
  - Auth: X-Api-Key
  - Optional: Set `rerank: null` in config to disable
  - Usage: Re-ranks retrieved chunks by relevance (retrieval_plugin: dense_rerank)

**Vision Providers (VLM):**
- **OpenAI GPT-4o** - Image understanding
  - SDK: `openai`
  - Auth: Bearer token
  - Usage: Describe figures/images during document parsing (docling_vision parser)
  - Optional: Set `vision: null` in config to disable (uses "[Figure]" placeholder)

- **Anthropic Claude Sonnet-4** - Image understanding
  - SDK: `anthropic`
  - Auth: Bearer token
  - Usage: Same as OpenAI for image description

- **Cohere** - Vision capabilities
  - SDK: `cohere`
  - Auth: X-Api-Key
  - Usage: Same as OpenAI for image description

## Data Storage

**Databases:**
- **PostgreSQL 14+** - Primary unified storage
  - Connection: Via psycopg (binary builds included)
  - Client: `psycopg[binary]>=3.1` with `psycopg-pool>=3.1` for pooling
  - Pool settings: Dynamic sizing, health checks on borrow
  - Extensions: `pgvector` for vector storage and operations
  - Modes:
    - **Local**: Embedded via `fitz-pgserver>=0.1.5` (fork of pgserver with Windows crash recovery)
      - Data: `~/.fitz/pgdata/`
      - Zero configuration, single `pip install fitz-pgserver`
    - **External**: User-provided PostgreSQL via `connection_string` in config
  - Storage: Vectors (pgvector), metadata (JSON), text tables, chunks
  - Indexing: HNSW (Hierarchical Navigable Small World) for 99% recall
  - Per-collection databases: Each collection is a separate PostgreSQL database

**Vector DB Features:**
- **pgvector plugin** (`fitz_ai/vector_db/plugins/pgvector.yaml`)
  - HNSW index with configurable parameters (hnsw_m, hnsw_ef_construction)
  - Hybrid search: Vector + full-text BM25 via tsvector
  - Full SQL capabilities: Aggregate, filter, join with metadata
  - Chunk storage: metadata includes summary, entities, content_type, hierarchy

**File Storage:**
- Local filesystem only - No external file storage integration (S3, etc.)
- Document ingestion: Files provided as local paths or URLs
- Source tracking: Original file path stored in chunk metadata (chunk.source.uri)

**Caching:**
- **Fitz Cloud Cache** (optional, `fitz_ai/cloud/`)
  - Service: https://api.fitz-ai.cloud/v1 (production) or http://localhost:8000/v1 (local development)
  - Purpose: Query-result caching with model routing recommendations
  - Auth: API key (`fitz_xxx` format) from Fitz Cloud dashboard
  - Encryption: AES-256-GCM client-side (org_key NEVER sent to server)
  - Config: `cloud.enabled`, `cloud.api_key`, `cloud.org_id`, `cloud.org_key`, `cloud.base_url`
  - Required env vars for testing: `FITZ_CLOUD_TEST_API_KEY`, `FITZ_CLOUD_TEST_ORG_KEY`, `FITZ_CLOUD_TEST_ORG_ID`
  - Features: Answer caching, model routing advice, deduplication hints
  - Tier limitations: Free tier cannot use cache, starter+ tier required

## Authentication & Identity

**Auth Provider:**
- Custom in-house implementation

**Authentication Mechanisms:**
- **API Key Auth** (`fitz_ai/llm/auth/api_key.py`)
  - Environment variables (lazy-loaded): Provider-specific var names
  - Header formats: Bearer (Authorization: Bearer xxx), X-Api-Key, Basic
  - No OAuth/OIDC integration

- **M2M Auth** (`fitz_ai/llm/auth/m2m.py`)
  - Machine-to-machine authentication for service-to-service calls
  - HTTP client with custom headers

**Environment Variables:**
- LLM providers use provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Cloud integration: FITZ_CLOUD_TEST_API_KEY, FITZ_CLOUD_TEST_ORG_KEY, FITZ_CLOUD_TEST_ORG_ID

## Monitoring & Observability

**Error Tracking:**
- Not detected - App-level error handling via FastAPI error handlers
- File: `fitz_ai/api/error_handlers.py`

**Logs:**
- Custom logging module: `fitz_ai/logging/logger.py`
- Log levels: DEBUG, INFO, WARNING, ERROR (configurable via `log_level` in config)
- Structured tags: STORAGE, RETRIEVAL, PARSING, CLOUD, etc. (`fitz_ai/logging/tags.py`)
- Output: Console by default (configurable)

**Distributed Tracing:**
- Not detected

## CI/CD & Deployment

**Hosting:**
- GitHub (source code repository)
- GitHub Actions (CI/CD pipeline)

**CI Pipeline:**
- **Lint & Type Check** (ubuntu-latest, Python 3.12)
  - Black (code formatter check)
  - isort (import sorting check)
  - Ruff (linter)

- **Unit Tests** (Python 3.10, 3.11, 3.12 on ubuntu-latest, windows-latest, macos-latest)
  - Tier 1 & 2: No external dependencies (mocked services)
  - `pytest tests/unit/`

- **Integration Tests** (Python 3.12 on ubuntu-latest)
  - Tier 3: Real PostgreSQL, optional Ollama
  - `pytest tests/integration/`

- **E2E Tests** (Python 3.12 on ubuntu-latest)
  - Tier 4: Real LLM APIs, document parsing, full pipeline
  - `pytest tests/e2e/` (marked with `@pytest.mark.e2e`)

- **Cloud Cache E2E Tests** (if env vars set)
  - Requires: FITZ_CLOUD_TEST_API_KEY, FITZ_CLOUD_TEST_ORG_KEY, FITZ_CLOUD_TEST_ORG_ID
  - File: `tests/integration/test_cloud_cache_e2e.py`

- **Mutation Testing** (nightly or manual)
  - Tool: `mutmut`
  - Target: `fitz_ai/core/` module
  - Command: `python -m pytest tests/unit/ tests/integration/ -x -q --tb=no`

**Deployment Target:**
- PyPI (Python package distribution)
- GitHub releases with version tags

**Release Automation:**
- GitHub Actions workflow: `.github/workflows/release.yml`
- Manual trigger or tag-based automation

## Environment Configuration

**Required Environment Variables:**
- LLM providers (at least one):
  - `ANTHROPIC_API_KEY` (for Anthropic)
  - `OPENAI_API_KEY` (for OpenAI)
  - `COHERE_API_KEY` (for Cohere)
  - None required for Ollama (local)

- Cloud cache (optional):
  - `FITZ_CLOUD_API_KEY` (Fitz Cloud dashboard)
  - `FITZ_CLOUD_ORG_ID` (or FITZ_ORG_ID)
  - `FITZ_CLOUD_ORG_KEY` (64-char hex, local encryption, never sent)

- Testing:
  - `FITZ_CLOUD_TEST_API_KEY` (for cloud cache E2E tests)
  - `FITZ_CLOUD_TEST_ORG_KEY` (64-char hex)
  - `FITZ_CLOUD_TEST_ORG_ID` (UUID format)

**Configuration Files:**
- Location: `~/.fitz/config/` (user home directory)
- Format: YAML
- Example: `~/.fitz/config/fitz_rag.yaml`
  ```yaml
  fitz_rag:
    chat: "anthropic/claude-sonnet-4"
    embedding: "openai/text-embedding-3-small"
    collection: "my_docs"
    vector_db_kwargs:
      mode: "local"  # or "external" with connection_string
  ```

**Secrets Location:**
- Environment variables (recommended)
- Config YAML (api_key, org_key fields)
- Never committed to git (use .env files locally)

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook receiving endpoints

**Outgoing:**
- Not detected - No outbound webhook sending

**API Endpoints:**
File: `fitz_ai/api/app.py` (FastAPI application)

REST Endpoints:
- `POST /ingest` - Ingest documents (`fitz_ai/api/routes/ingest.py`)
  - Request: IngestRequest (source path, collection, clear_existing flag)
  - Response: IngestResponse (documents, chunks, collection stats)

- `POST /query` - Query knowledge base (`fitz_ai/api/routes/query.py`)
  - Request: QueryRequest (query text, collection)
  - Response: QueryResponse (answer, sources, citations)

- `GET /health` - Health check (`fitz_ai/api/routes/health.py`)
  - Response: Health status

- `GET /collections` - List collections (`fitz_ai/api/routes/collections.py`)
  - Response: Available collections with stats

**HTTP Client:**
- Library: `httpx>=0.24` (async HTTP client)
- Used for: Cloud API calls, custom vector DB calls, provider communication

---

*Integration audit: 2026-01-30*
