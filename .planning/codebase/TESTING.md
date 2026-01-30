# Testing Patterns

**Analysis Date:** 2026-01-30

## Test Framework

**Runner:**
- Framework: **pytest** 7.0+
- Config: `pyproject.toml` `[tool.pytest.ini_options]`
- Test discovery:
  - Test files: `test_*.py` (e.g., `test_postgres_table_store.py`)
  - Test classes: `Test*` (e.g., `TestTableNameSanitization`)
  - Test functions: `test_*` (e.g., `test_simple_name_prefixed()`)
- Strict markers enforcement: Tests must declare markers they use

**Assertion Library:**
- Built-in `assert` statements (no external library needed)
- Useful patterns:
  ```python
  assert _sanitize_table_name("users") == "tbl_users"
  assert len(result) <= 63
  assert result.startswith("tbl_")
  assert isinstance(auth, AuthProvider)
  ```

**Run Commands:**
```bash
pytest                              # Run all tests
pytest -m tier1                     # Run tier1 (critical path, <30s)
pytest -m "tier1 or tier2"          # Run tiers 1+2 (PR merge, <2min)
pytest -m "tier1 or tier2 or tier3" # Run tiers 1+2+3 (merge to main, <10min)
pytest -m "not postgres"            # Skip postgres tests (for parallel runs)
pytest -k "test_simple"             # Run tests matching pattern
pytest tests/unit/test_postgres_table_store.py  # Run specific file
pytest --cov=fitz_ai                # Generate coverage report
pytest -xvs                         # Verbose, stop on first failure
pytest -n auto                      # Parallel execution (xdist)
```

## Test Tiers (CI/CD Optimization)

**Tier 1 - Critical Path (<30s):**
- Pure logic tests with no I/O or external dependencies
- No real services (postgres, APIs)
- No mocks for core functionality
- Run on every commit in CI
- Pattern files: `test_answer_mode.py`, `test_constraints.py`, `test_semantic_math.py`
- Location: `tests/unit/` with specific pattern matching in `conftest.py`

**Tier 2 - Unit with Mocks (<2min):**
- Tests with mocks for external services
- No real API calls (LLM, embeddings)
- PostgreSQL tests excluded (pgserver can't run in parallel)
- Run on PR merge in CI
- Includes: Most of `tests/unit/` except postgres-specific tests
- Pattern files: LLM client tests, integration adapter tests

**Tier 3 - Integration (<10min):**
- Real PostgreSQL (pgvector, table store)
- Real embedding/chat API calls (optional, skipped if not configured)
- End-to-end workflows
- Run on merge to main in CI
- Location: `tests/integration/`, `tests/e2e/`

**Tier 4 - Heavy (30min+):**
- Security tests (injection, leakage)
- Chaos/reliability tests (failure modes, recovery)
- Load/scalability tests (concurrent, large corpus)
- Performance benchmarks
- Run nightly or on-demand
- Location: `tests/security/`, `tests/chaos/`, `tests/load/`, `tests/performance/`

**Auto-Marking in conftest:**
- `tests/unit/conftest.py` automatically adds tier markers based on filename patterns
- `TIER1_PATTERNS`: List of test files that are pure logic
- `POSTGRES_PATTERNS`: List of test files that use PostgreSQL
- Postgres tests auto-skipped in parallel mode (pytest-xdist)

## Test File Organization

**Location:**
- **Unit tests:** `tests/unit/` (mirror fitz_ai/ structure where possible)
- **Integration tests:** `tests/integration/`
- **E2E tests:** `tests/e2e/` (real parser, retrieval, generation pipelines)
- **Property-based tests:** `tests/unit/property/` (hypothesis framework)
- **Chaos tests:** `tests/chaos/`
- **Performance tests:** `tests/performance/`
- **Security tests:** `tests/security/`
- **Load tests:** `tests/load/`

**Example structure:**
```
tests/
├── conftest.py                      # Root fixtures, tier marking, pgdata reset
├── test_config.yaml                 # Unified test config (embedders, chat)
├── unit/
│   ├── conftest.py                  # Mock embedders for unit tests
│   ├── mock_embedder.py             # Deterministic embedder for testing
│   ├── test_postgres_table_store.py # PostgreSQL table store tests
│   ├── llm/test_auth.py             # LLM auth unit tests
│   ├── property/                    # Property-based tests
│   │   ├── conftest.py
│   │   ├── strategies.py            # Hypothesis strategies
│   │   ├── test_chunkers.py
│   │   └── test_expansion_detector.py
│   └── ...
├── integration/
│   ├── conftest.py                  # Cloud fixtures
│   ├── cloud_fixtures.py            # Cloud-specific setup
│   └── test_cloud_cache_*.py        # Cloud integration tests
├── e2e/
│   ├── conftest.py
│   ├── test_retrieval_e2e.py        # Full RAG pipeline
│   ├── test_pdf_docx_retrieval.py   # Parser + retrieval
│   └── fixtures_rag/                # E2E test data
├── chaos/
│   ├── conftest.py
│   └── test_failure_modes.py
├── performance/
│   ├── conftest.py
│   └── test_*.py
└── load/
    ├── conftest.py
    └── locustfile.py
```

## Test Structure

**Fixture Pattern (pytest fixtures):**
```python
# tests/unit/conftest.py
@pytest.fixture
def mock_embedder():
    """Provides a deterministic mock embedder."""
    return create_deterministic_embedder(dimension=384)

@pytest.fixture
def semantic_matcher(mock_embedder):
    """Provides a SemanticMatcher with mock embedder."""
    return SemanticMatcher(embedder=mock_embedder)
```

**Test Class Organization:**
```python
# tests/unit/test_postgres_table_store.py

# =============================================================================
# Table Name Sanitization Tests
# =============================================================================

class TestTableNameSanitization:
    """Tests for table name sanitization."""

    def test_simple_name_prefixed(self):
        """Simple name gets tbl_ prefix."""
        assert _sanitize_table_name("users") == "tbl_users"
        assert _sanitize_table_name("orders") == "tbl_orders"

    def test_hyphens_replaced(self):
        """Hyphens are replaced with underscores."""
        assert _sanitize_table_name("my-table") == "tbl_my_table"


# =============================================================================
# Column Name Sanitization Tests
# =============================================================================

class TestColumnNameSanitization:
    """Tests for column name sanitization."""

    def test_simple_name_unchanged(self):
        """Simple alphanumeric name stays the same."""
        assert _sanitize_column_name("name") == "name"
```

**Patterns:**
- One test class per logical feature
- One assertion per test method (focused, easy to debug)
- Clear docstring describing what should happen
- Setup fixture dependencies as method arguments
- Cleanup handled by pytest automatically

**Async Testing:**
```python
# Using pytest-asyncio (if applicable)
@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operation."""
    result = await async_function()
    assert result == expected_value
```

**Error Testing:**
```python
# Test that specific exceptions are raised
def test_missing_env_var_raises(self):
    """Missing environment variable raises ValueError."""
    with patch.dict("os.environ", {}, clear=True):
        auth = ApiKeyAuth("NONEXISTENT_KEY")
        with pytest.raises(ValueError, match="NONEXISTENT_KEY not set"):
            auth.get_headers()

# Test exception messages
def test_query_error_message(self):
    """Query validation error includes field name."""
    with pytest.raises(QueryError, match="Query text cannot be empty"):
        Query(text="")
```

## Mocking

**Framework:** `unittest.mock` (standard library)

**Patterns:**

1. **Mock entire module:**
   ```python
   from unittest.mock import patch

   @patch("fitz_ai.llm.providers.cohere.ClientV2")
   def test_cohere_chat(self, mock_client_class):
       """Test Cohere chat with mocked client."""
       mock_client = MagicMock()
       mock_client_class.return_value = mock_client
       mock_client.chat.return_value = "Response"

       chat = get_chat("cohere")
       # ... test code
   ```

2. **Patch environment variables:**
   ```python
   with patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}):
       auth = ApiKeyAuth("COHERE_API_KEY")
       assert auth.get_headers()["Authorization"] == "Bearer test-key"
   ```

3. **Patch dependencies in conftest:**
   ```python
   # tests/unit/conftest.py
   @pytest.fixture
   def mock_embedder():
       """Provides a deterministic mock embedder."""
       from tests.unit.mock_embedder import create_deterministic_embedder
       return create_deterministic_embedder(dimension=384)
   ```

4. **Create test doubles for complex objects:**
   ```python
   from unittest.mock import MagicMock

   mock_chunk = MagicMock()
   mock_chunk.id = "chunk_1"
   mock_chunk.text = "Sample text"
   mock_chunk.metadata = {"source": "test.pdf"}
   ```

**What to Mock:**
- External services (LLM APIs, embedding services, cloud APIs)
- Database operations (when testing pure logic)
- Network calls (HTTP requests)
- File system operations (when not testing file handling)

**What NOT to Mock:**
- Core domain logic (Query, Answer, Chunk classes)
- Configuration objects
- Dataclass instances
- Core retrieval/generation algorithms (test with real logic)

## Fixtures and Factories

**Test Data Patterns:**

1. **Deterministic Embedder** (`tests/unit/mock_embedder.py`):
   ```python
   def create_deterministic_embedder(dimension: int = 384) -> Callable[[str], list[float]]:
       """Create a mock embedder with semantic clusters."""
       # Maps similar-meaning texts to same vector clusters
       # Maintains cosine similarity > 0.70 within clusters
   ```
   - Used for: Testing semantic matching, guardrails, detection modules
   - Benefit: No API calls, deterministic results, fast execution
   - Example vectors:
     ```python
     CAUSAL_QUERY_CLUSTER      # "why", "explain", "what caused"
     FACT_QUERY_CLUSTER        # "what", "who", "where", "when"
     SUCCESS_CLUSTER           # "successful", "completed"
     FAILURE_CLUSTER           # "failed", "rejected"
     ```

2. **Test Config** (`tests/test_config.yaml`):
   ```yaml
   vector_db: pgvector
   tiers:
     - name: local
       chat: ollama
       chat_kwargs:
         models: {smart: qwen2.5:7b, fast: qwen2.5:1.5b}
       embedding: ollama
       embedding_kwargs:
         model: nomic-embed-text
   ```
   - Centralized for all tests
   - Loaded once per session and cached
   - Available via `load_test_config()` in `tests/conftest.py`

3. **Cloud Fixtures** (`tests/integration/cloud_fixtures.py`):
   - `cloud_available`: Skip if cloud credentials missing
   - `cloud_client`: CloudClient instance
   - `cloud_config`: CloudConfig from env vars
   - `cloud_pipeline`: FitzRagEngine with cloud enabled
   - Require env vars: `FITZ_CLOUD_TEST_API_KEY`, `FITZ_CLOUD_TEST_ORG_KEY`, etc.

4. **Property-Based Strategies** (`tests/unit/property/strategies.py`):
   ```python
   from hypothesis import strategies as st

   document_text = st.text(
       alphabet=st.characters(blacklist_categories=("Cc", "Cs")),
       min_size=10,
       max_size=1000,
   )

   chunk_params = st.fixed_dictionaries({
       "min_length": st.integers(min_value=100, max_value=500),
       "max_length": st.integers(min_value=500, max_value=2000),
   })
   ```
   - Used in: `tests/unit/property/test_chunkers.py`, etc.
   - Framework: hypothesis (property-based testing)

**Location:**
- `tests/unit/mock_embedder.py`: Deterministic embedder
- `tests/conftest.py`: Root fixtures (config loading, pgdata cleanup)
- `tests/unit/conftest.py`: Unit test fixtures (mock embedders)
- `tests/e2e/conftest.py`: E2E fixtures (real embedders, documents)
- `tests/integration/cloud_fixtures.py`: Cloud-specific fixtures
- `tests/unit/property/strategies.py`: Hypothesis strategies

## Coverage

**Requirements:** No explicit coverage threshold enforced (pragmatic)

**View Coverage:**
```bash
pytest --cov=fitz_ai --cov-report=html
pytest --cov=fitz_ai --cov-report=term-missing
```

**Focus Areas:**
- Core exception handling (QueryError, KnowledgeError paths)
- Query validation (empty text, invalid constraints)
- Retrieval intelligence modules (temporal, aggregation, comparison)
- Configuration loading and validation
- Cloud cache operations (hit/miss paths)
- Error recovery (postgres crash recovery, pgserver cleanup)

**Known Gaps:**
- Cloud integration (requires test credentials)
- LLM provider-specific error handling (requires real APIs)
- Load/scalability edge cases (requires test infrastructure)

## Test Types

**Unit Tests:**
- Scope: Single function or class in isolation
- Location: `tests/unit/`
- Mocks: External services, file I/O
- Speed: <1ms per test
- Example: `TestTableNameSanitization`, `TestApiKeyAuth`

**Integration Tests:**
- Scope: Multiple components working together
- Location: `tests/integration/`
- Real services: PostgreSQL, cloud APIs
- Speed: 100ms - 10s per test
- Example: Cloud cache integration with FitzRagEngine

**E2E Tests:**
- Scope: Full pipeline (ingest → retrieve → generate)
- Location: `tests/e2e/`
- Real services: Document parser, embeddings, LLM, vector DB
- Speed: 1s - 30s per test
- Markers: `@pytest.mark.e2e`, `@pytest.mark.llm`, `@pytest.mark.embeddings`
- Example: `test_retrieval_e2e.py`, `test_pdf_docx_retrieval.py`

**Property-Based Tests:**
- Framework: **hypothesis**
- Scope: Properties that should hold for all inputs
- Location: `tests/unit/property/`
- Example: "Chunker should never return empty chunks"
- Marker: `pytestmark = pytest.mark.property`

**Performance Tests:**
- Scope: Benchmarking specific operations
- Location: `tests/performance/`
- Marker: `@pytest.mark.performance`
- Example: Measure retrieval latency under load

**Security Tests:**
- Scope: Injection attacks, leakage, validation
- Location: `tests/security/`
- Marker: `@pytest.mark.security`
- Example: SQL injection prevention in tabular store

**Chaos Tests:**
- Scope: Failure modes, recovery, reliability
- Location: `tests/chaos/`
- Marker: `@pytest.mark.chaos`
- Example: PostgreSQL crash recovery, network failures

## Common Patterns

**Async Testing (if needed):**
```python
import pytest

@pytest.mark.asyncio
async def test_async_retrieval():
    """Test asynchronous retrieval."""
    engine = await create_engine_async()
    result = await engine.retrieve("query")
    assert len(result) > 0
```

**Error Testing:**
```python
def test_invalid_query_raises(self):
    """Empty query raises QueryError."""
    with pytest.raises(QueryError, match="cannot be empty"):
        Query(text="")

def test_cloud_disabled_by_default(self):
    """Cloud is not enabled if config missing."""
    config = FitzRagConfig(cloud=None)
    engine = FitzRagEngine(config)
    assert engine._cloud_client is None
```

**Mocking External APIs:**
```python
from unittest.mock import patch, MagicMock

@patch("fitz_ai.llm.providers.cohere.ClientV2")
def test_cohere_embedding(self, mock_client_class):
    """Test embedding with mocked Cohere API."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.embed_api.embed.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }

    embedder = get_embedder("cohere")
    result = embedder.embed("test text")
    assert len(result[0]) == 3
```

**Dependency Injection in Tests:**
```python
def test_pipeline_with_mock_chat(self):
    """Pipeline uses injected chat client."""
    mock_chat = MagicMock()
    mock_chat.chat.return_value = "Test answer"

    pipeline = RAGPipeline(config=config, chat=mock_chat)
    answer = pipeline.generate(chunks=[])
    assert answer == "Test answer"
```

**Fixtures with Cleanup:**
```python
@pytest.fixture
def temp_database(tmp_path):
    """Create temporary database and clean up."""
    db_path = tmp_path / "test.db"
    db = DatabaseConnection(db_path)
    yield db  # Test runs here
    db.close()  # Cleanup runs after test
```

## Markers and Tags

**Tier Markers** (CI/CD control):
```bash
pytest -m tier1              # Critical path (<30s)
pytest -m "tier1 or tier2"   # Unit tests with mocks (<2min)
pytest -m "not tier4"        # Skip heavy tests
```

**Feature Markers** (skip optional features):
```bash
pytest -m "not llm"          # Skip tests requiring real LLM
pytest -m "not embeddings"   # Skip tests requiring real embeddings
pytest -m "not postgres"     # Skip PostgreSQL tests (for parallel)
pytest -m "not slow"         # Skip slow tests (>10s)
```

**Category Markers** (test type):
```bash
pytest -m integration        # Integration tests only
pytest -m e2e                # End-to-end tests
pytest -m security           # Security tests
pytest -m chaos              # Chaos/reliability tests
pytest -m performance        # Performance benchmarks
pytest -m scalability        # Scalability tests
pytest -m property           # Property-based tests
```

## Known Limitations

1. **PostgreSQL Parallelization:**
   - pgserver (embedded PostgreSQL) cannot run in parallel with pytest-xdist
   - Auto-skips postgres tests when running with `-n auto`
   - Run postgres tests separately: `pytest -m postgres`

2. **Cloud Tests:**
   - Require environment variables for credentials
   - Skip silently if not configured
   - Manual testing recommended for cloud features

3. **LLM/Embedding Tests:**
   - Can be slow and unreliable (dependent on external APIs)
   - Marked with `@pytest.mark.llm` and `@pytest.mark.embeddings`
   - Mock versions available in `tests/unit/` for faster tests

---

*Testing analysis: 2026-01-30*
