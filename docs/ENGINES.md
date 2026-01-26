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
- Vector database for storage (Qdrant, FAISS)
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

## Fitz RAG Engine

**Location**: `fitz_ai/engines/fitz_rag/`

### Usage

```python
from fitz_ai.engines.fitz_rag import run_fitz_rag

answer = run_fitz_rag("What is quantum computing?")
print(answer.text)
print(answer.provenance)
```

### Configuration

```yaml
# .fitz/config/fitz_rag.yaml
chat:
  plugin_name: cohere
  kwargs:
    models:
      smart: command-a-03-2025
      fast: command-r7b-12-2024

embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

vector_db: pgvector
vector_db_kwargs:
  mode: local  # or "external" with connection_string

# Retrieval strategy - plugin choice controls reranking
retrieval:
  plugin_name: dense_rerank  # or "dense" for no reranking
  collection: my_knowledge
  top_k: 5

# Rerank provider (used only if retrieval.plugin_name includes reranking)
rerank:
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5
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
from fitz import run

# Default: Fitz RAG
answer = run("What is X?", engine="fitz_rag")

# Custom engine (see CUSTOM_ENGINES.md)
answer = run("What is X?", engine="my_custom_engine")
```

### Available Engines

| Engine | Description | Status |
|--------|-------------|--------|
| `fitz_rag` | Traditional RAG with epistemic guardrails | Production |

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

## Architecture Principles

1. **Protocol-Based**: Engines implement protocols, not inherit from base classes
2. **Config-Driven**: Engine behavior controlled by configuration
3. **Shared Infrastructure**: LLM, vector DB, ingestion shared across engines
4. **Standardized I/O**: All engines use Query → Answer
