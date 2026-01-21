# Python SDK Reference

Complete reference for the Fitz Python SDK.

---

## Quick Start

```python
import fitz_ai

# Ingest documents
fitz_ai.ingest("./docs")

# Ask questions
answer = fitz_ai.query("What is the refund policy?")
print(answer.text)
```

---

## Module-Level API

The simplest way to use Fitz - matches CLI behavior.

### fitz_ai.ingest()

Ingest documents into the knowledge base.

```python
fitz_ai.ingest(
    source,                    # Path to file or directory
    collection: str = None,    # Collection name (uses default)
    clear_existing: bool = False  # Clear collection first
) -> IngestStats
```

**Returns:** `IngestStats` with `documents`, `chunks`, `collection`

**Examples:**

```python
# Basic ingestion
fitz_ai.ingest("./docs")

# With collection name
fitz_ai.ingest("./physics_papers", collection="physics")

# Clear and re-ingest
fitz_ai.ingest("./docs", clear_existing=True)
```

### fitz_ai.query()

Query the knowledge base.

```python
fitz_ai.query(
    question: str,             # The question to ask
    top_k: int = None          # Override chunk count
) -> Answer
```

**Returns:** `Answer` with `text`, `provenance`, `mode`

**Examples:**

```python
answer = fitz_ai.query("What is the refund policy?")
print(answer.text)
print(answer.mode)  # CONFIDENT, QUALIFIED, DISPUTED, or ABSTAIN

# Access sources
for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

---

## fitz Class

For advanced usage with multiple collections or custom configuration.

### Constructor

```python
from fitz_ai import fitz

f = fitz(
    collection: str = "default",     # Collection name
    config_path: str = None,         # Custom config file
    auto_init: bool = True           # Create config if missing
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | str | `"default"` | Vector DB collection name |
| `config_path` | str/Path | None | Path to YAML config |
| `auto_init` | bool | True | Create default config if missing |

### Methods

#### ingest()

```python
f.ingest(
    source: str | Path,
    clear_existing: bool = False
) -> IngestStats
```

#### ask() / query()

```python
f.ask(
    question: str,
    top_k: int = None
) -> Answer
```

Note: `query()` is an alias for `ask()`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `collection` | str | The collection name |
| `config_path` | Path | Path to config file |

### Examples

```python
from fitz_ai import fitz

# Multiple collections
physics = fitz(collection="physics")
physics.ingest("./physics_papers")
physics_answer = physics.ask("Explain entanglement")

legal = fitz(collection="legal")
legal.ingest("./contracts")
legal_answer = legal.ask("What are the payment terms?")

# Custom config
f = fitz(config_path="./my_config.yaml")

# Require existing config
f = fitz(auto_init=False)  # Raises if no config
```

---

## Core Types

### Answer

The response from a query.

```python
from fitz_ai import Answer

class Answer:
    text: str                    # The answer text
    provenance: list[Provenance] # Sources used
    mode: AnswerMode | None      # Epistemic mode
    metadata: dict               # Additional data
```

**Answer Modes:**

| Mode | Description |
|------|-------------|
| `CONFIDENT` | Strong evidence supports the answer |
| `QUALIFIED` | Answer with caveats/limitations |
| `DISPUTED` | Conflicting sources detected |
| `ABSTAIN` | Insufficient evidence to answer |

### Provenance

Source attribution for an answer.

```python
from fitz_ai import Provenance

class Provenance:
    source_id: str    # Unique source identifier
    excerpt: str      # Relevant excerpt
    metadata: dict    # Additional source info
```

### IngestStats

Statistics from ingestion.

```python
from fitz_ai import IngestStats

class IngestStats:
    documents: int   # Number of documents ingested
    chunks: int      # Number of chunks created
    collection: str  # Target collection name
```

### Query

Input to the engine (for advanced usage).

```python
from fitz_ai import Query

query = Query(
    text="What is X?",
    constraints=Constraints(max_sources=5),
    metadata={"user_id": "123"}
)
```

### Constraints

Query-time constraints (for advanced usage).

```python
from fitz_ai import Constraints

constraints = Constraints(
    max_sources: int = None,     # Limit source count
    require_grounding: bool = True,  # Must be grounded
    metadata: dict = None        # Additional constraints
)
```

---

## Advanced Usage

### Direct Engine Access

```python
from fitz_ai import create_engine, Query

# Create engine instance
engine = create_engine("fitz_rag")

# Build query
query = Query(text="What is X?")

# Get answer
answer = engine.answer(query)
```

### Engine Selection

```python
from fitz_ai import run, list_engines

# List available engines
engines = list_engines()
print(engines)  # ['fitz_rag']

# Run with specific engine
answer = run("What is X?", engine="fitz_rag")
```

### Fitz RAG Specific

```python
from fitz_ai import run_fitz_rag, create_fitz_rag_engine

# RAG-specific entry point
answer = run_fitz_rag("What is X?")

# Create reusable RAG engine
engine = create_fitz_rag_engine()
```

---

## Error Handling

```python
from fitz_ai import (
    ConfigurationError,
    EngineError,
    QueryError,
    KnowledgeError,
    GenerationError,
)

try:
    answer = fitz_ai.query("What is X?")
except ConfigurationError as e:
    print(f"Config issue: {e}")
except QueryError as e:
    print(f"Query failed: {e}")
except EngineError as e:
    print(f"Engine error: {e}")
```

| Exception | When |
|-----------|------|
| `ConfigurationError` | Config file missing or invalid |
| `QueryError` | Invalid query or retrieval failed |
| `EngineError` | Engine initialization or execution error |
| `GenerationError` | LLM generation failed |
| `KnowledgeError` | Base class for knowledge errors |

---

## Configuration

The SDK uses the same config as CLI. See [CONFIG.md](CONFIG.md) for details.

**Config search order:**
1. `config_path` parameter (if provided)
2. `.fitz/config/fitz_rag.yaml` (project config)
3. Auto-created default config (if `auto_init=True`)

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [API.md](API.md) - REST API documentation
- [INGESTION.md](INGESTION.md) - Ingestion pipeline details
