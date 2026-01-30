# Coding Conventions

**Analysis Date:** 2026-01-30

## Naming Patterns

**Files:**
- Module files: `snake_case` (e.g., `vector_search.py`, `pgvector.py`)
- Test files: `test_<module_name>.py` (e.g., `test_postgres_table_store.py`)
- Config files: `<name>.py` or `<name>.yaml` (e.g., `schema.py`, `config.yaml`)
- Directories: `snake_case` (e.g., `fitz_ai/engines/fitz_rag/retrieval/steps/`)

**Classes:**
- Classes: `PascalCase` (e.g., `FitzRagEngine`, `VectorSearch`, `CloudClient`)
- Abstract base classes with `ABC` inheritance or `@abstractmethod` decorators
- Protocol classes with `@runtime_checkable` decorator for structural typing
- Example: `KnowledgeEngine` (protocol), `DetectionModule` (ABC), `FitzRagEngine` (concrete)

**Functions:**
- Functions: `snake_case` (e.g., `get_logger()`, `_sanitize_table_name()`)
- Private functions: `_leading_underscore()` (e.g., `_force_remove_pgdata()`)
- Factory functions: `from_config()`, `create_*()` (e.g., `RAGPipeline.from_config()`)

**Variables:**
- Variables: `snake_case` (e.g., `query_text`, `embedding_dimension`, `max_hops`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_MODULES`, `TIER1_PATTERNS`)
- Private attributes: `_leading_underscore` (e.g., `self._config`, `self._pipeline`)
- Type variables and generics: Single uppercase letter or descriptive PascalCase (e.g., `T`, `ResultType`)

**Enums:**
- Enum names: `PascalCase` (e.g., `DetectionCategory`, `TemporalIntent`, `ElementType`)
- Enum values: `UPPER_SNAKE_CASE` (e.g., `DetectionCategory.TEMPORAL`, `TemporalIntent.PAST_TENSE`)

## Code Style

**Formatting:**
- Tool: **Black** (enforced in CI)
- Line length: **100 characters**
- Target versions: Python 3.10, 3.11, 3.12
- Config: `pyproject.toml` `[tool.black]` section

**Import Organization:**
- Tool: **isort** (enforced in CI)
- Profile: **black** (compatible with Black)
- Line length: **100 characters**
- Order (enforced by isort):
  1. Standard library imports (e.g., `import os`, `from pathlib import Path`)
  2. Third-party imports (e.g., `import pydantic`, `import httpx`)
  3. Local imports (e.g., `from fitz_ai.core import Query`, `from .protocol import DetectionResult`)
- Example from `fitz_ai/engines/fitz_rag/engine.py`:
  ```python
  from __future__ import annotations

  import os
  from typing import TYPE_CHECKING

  from fitz_ai.cloud import CloudClient
  from fitz_ai.core import (
      Answer,
      ConfigurationError,
      GenerationError,
      KnowledgeError,
      Provenance,
      Query,
      QueryError,
  )
  from fitz_ai.engines.fitz_rag.config.schema import FitzRagConfig
  from fitz_ai.engines.fitz_rag.generation.retrieval_guided.synthesis import RGSAnswer
  from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline
  from fitz_ai.logging.logger import get_logger

  if TYPE_CHECKING:
      pass
  ```

**Linting:**
- Tool: **ruff** (enforced in CI)
- Line length: **100 characters**
- Per-file exceptions in `pyproject.toml`:
  - `tools/contract_map/`: E402 (import after statements allowed)
  - `tests/**`: E402 (pytest markers before imports)
  - `examples/**`: E402 (contextual imports for teaching)

**Type Hints:**
- Required for all public APIs (functions, methods, class attributes)
- Recommended for private functions and variables
- Use `from __future__ import annotations` at top of every file for forward references (259+ files do this)
- Use `TYPE_CHECKING` guard for expensive imports only needed in type checkers:
  ```python
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from some_expensive_module import HeavyClass
  ```

**Docstrings:**
- Style: **Google** format (3-line minimum)
- Location: Module docstring at top, docstrings for all public classes/functions
- Content:
  - One-line summary (present tense)
  - Blank line
  - Multi-paragraph description (if needed)
  - Args section (if function has parameters)
  - Returns section (if function returns value)
  - Raises section (if function raises exceptions)
  - Examples section (for important/complex functions)
- Example from `fitz_ai/engines/fitz_rag/engine.py`:
  ```python
  class FitzRagEngine:
      """
      Fitz RAG engine implementation.

      This engine implements the retrieval-augmented generation paradigm:
      1. Embed the query
      2. Retrieve relevant chunks from vector DB
      3. Optionally rerank chunks
      4. Generate answer using LLM + retrieved context

      The engine wraps the existing RAGPipeline and adapts it to the
      KnowledgeEngine protocol.

      Examples:
          >>> from fitz_ai.config import load_engine_config
          >>>
          >>> config = load_engine_config("fitz_rag")
          >>> engine = FitzRagEngine(config)
          >>>
          >>> query = Query(text="What is quantum computing?")
          >>> answer = engine.answer(query)
          >>> print(answer.text)
          >>> for source in answer.provenance:
          ...     print(f"Source: {source.source_id}")
      """
  ```

## Import Organization

**Path Aliases:**
- No import path aliases currently used
- Always use absolute imports from package root: `from fitz_ai.core import Query`
- Never use relative imports in shared code (only within isolated components)

**Circular Import Prevention:**
- Use `TYPE_CHECKING` guards for type-only imports between modules
- Use late binding in function bodies if needed to break cycles
- Layer enforcement via `tools/contract_map/` to prevent architecture violations

**Barrel Files:**
- Used selectively in `__init__.py` files for public API exposure
- Example: `fitz_ai/core/__init__.py` exports main classes:
  ```python
  from .answer import Answer
  from .engine import KnowledgeEngine
  from .exceptions import EngineError, GenerationError, KnowledgeError, QueryError
  from .query import Query
  ```

## Error Handling

**Exception Hierarchy:**
- Base: `EngineError` (parent of all engine errors)
- Query validation: `QueryError` (empty text, invalid constraints, malformed metadata)
- Knowledge access: `KnowledgeError` (vector DB failures, missing docs, corrupted index)
- Generation: `GenerationError` (LLM failures, timeout, invalid response)
- Configuration: `ConfigurationError` (invalid config, missing components)
- Specialized: `StorageError`, `EmbeddingError`, etc. (inherit from `EngineError`)

**Raising Exceptions:**
- Always include meaningful context message
- Chain exceptions with `from e` to preserve stack trace:
  ```python
  try:
      self._pipeline = RAGPipeline.from_config(config, cloud_client=self._cloud_client)
  except Exception as e:
      raise ConfigurationError(f"Failed to initialize Fitz RAG engine: {e}") from e
  ```
- Distinguish between user errors (e.g., `QueryError`) and system errors (e.g., `KnowledgeError`)

**Validation:**
- Validate inputs early at function entry points
- Return helpful error messages with actual vs. expected values
- Example from `FitzRagEngine.answer()`:
  ```python
  if not query.text or not query.text.strip():
      raise QueryError("Query text cannot be empty")
  ```

## Logging

**Framework:** Python `logging` module (standard library)

**Setup:**
- Centralized in `fitz_ai/logging/logger.py`
- Called once early in lifecycle (CLI entrypoint calls `configure_logging()`)
- Safe to call multiple times (handler duplication prevented)

**Usage Pattern:**
```python
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)  # Automatic namespace from module path
logger.info("Message here", extra={"key": "value"})
```

**Log Levels:**
- `DEBUG`: Development-only diagnostic information
- `INFO`: Significant events (component startup, feature enabled)
- `WARNING`: Recoverable issues (fallback triggered, retry attempt)
- `ERROR`: Failures requiring attention (failed operation, invalid state)

**Logging Guidelines:**
- Use structured logging with `extra={}` dict for context:
  ```python
  logger.info("Fitz Cloud enabled", extra={"org_id": org_id[:8]})
  ```
- Log at component boundaries (initialization, major state changes)
- Never log sensitive data (API keys, credentials, PII)
- Don't use `f-strings` in format—use `extra` dict for values:
  ```python
  logger.info("Processing", extra={"count": count})  # Good
  logger.info(f"Processing {count}")  # Bad - string formatting
  ```

## Comments

**When to Comment:**
- Explain *why* a decision was made, not *what* the code does
- Document non-obvious algorithm choices or workarounds
- Mark temporary fixes or known limitations with `TODO` or `FIXME`
- Explain complex logic, especially in detection modules or enrichment pipeline

**JSDoc/TSDoc (Python equivalent - docstrings):**
- Use Google-style docstrings for all public APIs
- Include Args, Returns, Raises sections for clarity
- Use example sections for complex or frequently-used functions
- No inline `# type: int` comments—use type hints in signatures instead

**Code Comments vs. Docstrings:**
- Docstrings: For public APIs (modules, classes, functions)
- Comments: For complex logic within functions, design decisions, workarounds
- Example:
  ```python
  def retrieve(self, query: str) -> list[Chunk]:
      """Retrieve relevant chunks for query."""
      # Use RRF (reciprocal rank fusion) to combine sparse + dense scores
      # This ensures both keyword and semantic relevance
      sparse_results = self._sparse_search(query)
      dense_results = self._dense_search(query)
      return self._rrf_combine(sparse_results, dense_results)
  ```

## Function Design

**Size Guidelines:**
- Prefer functions under 50 lines (split complex logic into helpers)
- Methods in classes: Under 30 lines typical
- Orchestration/factory functions: Can be longer if each section is clear

**Parameters:**
- Prefer concrete types to unions (use overloads if needed)
- Max 5 positional parameters (use dataclass/NamedTuple for more)
- Optional parameters with defaults should come after required ones
- Example:
  ```python
  def execute(
      self,
      query: str,
      chunks: list[Chunk],
      k: int = 10,
      detection: DetectionSummary | None = None,
  ) -> list[Chunk]:
      """Execute retrieval strategy."""
  ```

**Return Values:**
- Return concrete types, not unions (except Optional)
- Use dataclasses for multiple return values:
  ```python
  @dataclass
  class RetrievalResult:
      chunks: list[Chunk]
      confidence: float
      took_ms: int
  ```
- Single return value per path (avoid multiple return types in one function)

**Dependencies:**
- Use dependency injection (pass services as parameters)
- Avoid global state except for logger
- Factory methods (`from_config()`) handle complex initialization

## Module Design

**Exports:**
- Public API goes in module docstring and `__all__` if list is short
- Use `__init__.py` to expose primary interfaces:
  ```python
  from .answer import Answer
  from .engine import KnowledgeEngine
  from .query import Query

  __all__ = ["Answer", "KnowledgeEngine", "Query"]
  ```

**Barrel Files:**
- Used in `fitz_ai/core/__init__.py`, `fitz_ai/llm/__init__.py` for public APIs
- Keep module hierarchy visible (don't hide deep nesting)
- Example good structure:
  - `fitz_ai/core/query.py` (implementation)
  - `fitz_ai/core/__init__.py` (exports Query)
  - Usage: `from fitz_ai.core import Query` (clean import)

**Module Organization:**
- One class per file for complex classes (e.g., `vector_search.py` = `VectorSearch` class)
- Group related utilities in files (e.g., `sanitizers.py` = multiple `_sanitize_*()` functions)
- Protocols in `protocol.py` files with dataclass companions
- Factories in `*_factory.py` or `create_*()` functions

## Patterns Used Throughout

**Protocol-Based Design:**
- Used for polymorphism without inheritance (e.g., `KnowledgeEngine`, `AuthProvider`)
- Benefits: Clear contracts, no forced inheritance, supports duck typing

**Dataclass-Heavy:**
- Use `@dataclass` for data containers (most config, results, DTOs)
- Use `frozen=True` for immutable values
- Example: `DetectionResult`, `DetectionSummary`, `Answer`

**Factory Methods:**
- `from_config()` pattern for complex initialization from config objects
- `create_*()` functions for service instantiation
- Example: `RAGPipeline.from_config(config, cloud_client=None)`

**Builder/Fluent:**
- Not commonly used; prefer configuration objects instead

**Dependency Injection:**
- Explicitly passed as constructor parameters or method arguments
- No service locators or global registries (except logger)
- Makes testing easier (inject mocks)

---

*Convention analysis: 2026-01-30*
