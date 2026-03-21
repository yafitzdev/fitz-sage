# Fitz RAG Engine

This document explains Fitz's RAG engine architecture and core contracts.

---

## Overview

Fitz RAG is a traditional Retrieval-Augmented Generation engine with epistemic guardrails and hierarchical summarization capabilities.

```
Query → Embed → Vector Search → Rerank → Context Build → LLM → Answer
```

**Key Features**:
- Separate embedding model for retrieval
- Vector database for storage (pgvector)
- Chunk-based retrieval with optional reranking
- Hierarchical summaries for analytical queries
- Epistemic guardrails (knows when to say "I don't know")

---

## Core Contracts

All engines implement the same interface, defined in `fitz_ai/core/`:

### KnowledgeEngine Protocol

```python
from typing import Protocol
from fitz_ai.core import Query, Answer

class KnowledgeEngine(Protocol):
    """Protocol that all knowledge engines must implement."""

    def answer(self, query: Query) -> Answer:
        """Execute a query and return an answer."""
        ...
```

### Query

```python
@dataclass
class Query:
    text: str                           # The question
    constraints: Optional[Constraints]  # Query-time limits
    metadata: Dict[str, Any]            # Engine hints
```

### Answer

```python
@dataclass
class Answer:
    text: str                      # Generated answer
    provenance: List[Provenance]   # Source attribution
    metadata: Dict[str, Any]       # Engine metadata
```

### Provenance

```python
@dataclass
class Provenance:
    source_id: str              # Unique source identifier
    excerpt: Optional[str]      # Relevant excerpt
    metadata: Dict[str, Any]    # Source metadata
```

---

## Fitz KRAG Engine

**Location**: `fitz_ai/engines/fitz_krag/`

### Usage

```python
from fitz_ai.engines.fitz_krag import run_fitz_krag

answer = run_fitz_krag("What is quantum computing?")
print(answer.text)
print(answer.provenance)
```

### Configuration

```yaml
# .fitz/config.yaml
chat_fast: cohere/command-r7b-12-2024
chat_balanced: cohere/command-r-08-2024
chat_smart: cohere/command-a-03-2025
embedding: cohere/embed-v4.0
rerank: cohere/rerank-v3.5       # or null to disable
collection: my_knowledge
parser: glm_ocr                  # or docling, docling_vision

vector_db: pgvector
vector_db_kwargs:
  mode: local  # or "external" with connection_string
```

### Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Summaries** | L0 chunks + L1 doc summaries + L2 corpus summary |
| **Epistemic Guardrails** | Detects contradictions, insufficient evidence |
| **Artifact Generation** | Auto-generates architecture docs, data models, etc. |
| **Incremental Ingestion** | Only re-processes changed files |

---

## Alternative Engines

Fitz supports a pluggable engine architecture. You can swap engines via configuration:

```python
from fitz_ai import run

# Default: Fitz KRAG
answer = run("What is X?", engine="fitz_krag")

# Custom engine (see CUSTOM_ENGINES.md)
answer = run("What is X?", engine="my_custom_engine")
```

### Available Engines

| Engine | Description | Status |
|--------|-------------|--------|
| `fitz_krag` | KRAG with epistemic guardrails | Production |

Custom engines can be registered via the engine registry. See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md).

---

## Custom Engines

You can create your own engine. See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md) for details.

```python
from fitz_ai.core import Query, Answer
from fitz_ai.runtime import EngineRegistry

class MyEngine:
    def answer(self, query: Query) -> Answer:
        # Your logic here
        return Answer(text="...", provenance=[])

# Register and use
EngineRegistry.get_global().register("my_engine", lambda c: MyEngine())
```

---

## Standalone Code Retrieval

For code-only retrieval without the full KRAG stack, fitz-ai provides a lightweight `CodeRetriever`:

```bash
pip install fitz-ai[code]
```

```python
from fitz_ai.code import CodeRetriever

retriever = CodeRetriever(source_dir="./myproject", chat_factory=my_factory)
results = retriever.retrieve("How does authentication work?")
```

**Pipeline:** AST structural index → LLM file selection → import graph expansion → neighbor expansion → compression.

No PostgreSQL, no pgvector, no docling. See [KRAG docs](features/platform/krag.md#standalone-code-retrieval) for details.

| Component | Path |
|-----------|------|
| CodeRetriever | `fitz_ai/code/retriever.py` |
| Indexer (file list, AST index, import graph) | `fitz_ai/code/indexer.py` |
| LLM prompts | `fitz_ai/code/prompts.py` |

---

## Architecture Principles

1. **Protocol-Based**: Engines implement protocols, not inherit from base classes
2. **Config-Driven**: Engine behavior controlled by configuration
3. **Shared Infrastructure**: LLM, vector DB, ingestion shared across engines
4. **Standardized I/O**: All engines use Query → Answer
