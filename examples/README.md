# Fitz Examples

This directory contains example scripts demonstrating Fitz usage.

## Python SDK

The simplest way to use Fitz programmatically:

```python
from fitz_ai.sdk import fitz

# Create a Fitz instance
f = fitz(collection="my_docs")

# Ingest documents
f.ingest("./docs")

# Ask questions
answer = f.ask("What is the refund policy?")

print(answer.text)
for source in answer.provenance:
    print(f"  - {source.source_id}")
```

## Examples

| File | Description |
|------|-------------|
| `quickstart.py` | Engine API usage with `run_fitz_rag` and `Query` objects |
| `ingestion_example.py` | Document ingestion and chunking flow |
| `universal_cli.py` | Building a CLI with the universal runtime API |

## Running Examples

### Quickstart

```bash
# Requires: fitz init (or existing config)
python examples/quickstart.py
```

Demonstrates:
- Simple one-off queries with `run_fitz_rag()`
- Query constraints
- Reusable engine instances
- Error handling
- Working with Answer objects

### Ingestion Example

```bash
# Create test documents first
mkdir -p test_docs
echo "RAG systems combine retrieval with generation." > test_docs/pipeline.txt
echo "Vector databases enable semantic search." > test_docs/vectors.txt

# Run the example
python examples/ingestion_example.py
```

Demonstrates:
- Reading documents from filesystem
- Validating documents
- Chunking strategies

### Universal CLI

```bash
python examples/universal_cli.py --help
python examples/universal_cli.py query "What is RAG?"
python examples/universal_cli.py engines --verbose
```

Demonstrates:
- Building CLI tools with `fitz_ai.runtime`
- Listing available engines
- Running queries with different engines

## Quick Patterns

### Using the Runtime API

```python
from fitz_ai.runtime import run, list_engines

# List available engines
print(list_engines())  # ['fitz_rag']

# Run a query
answer = run("What is quantum computing?", engine="fitz_rag")
print(answer.text)
```

### Using the Engine Directly

```python
from fitz_ai.engines.fitz_rag import run_fitz_rag, create_fitz_rag_engine
from fitz_ai.core import Query, Constraints

# One-off query
answer = run_fitz_rag("What is X?")

# Reusable engine
engine = create_fitz_rag_engine("config.yaml")
query = Query(text="Explain Y", constraints=Constraints(max_sources=5))
answer = engine.answer(query)
```

### Using the SDK

```python
from fitz_ai.sdk import fitz

f = fitz(collection="my_knowledge")
f.ingest("./documents")
answer = f.ask("What are the key findings?")
```
