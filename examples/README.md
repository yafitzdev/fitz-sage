# Fitz Examples

Practical, copy-paste-ready examples for common use cases.

## Quick Start

```bash
pip install fitz-ai
export COHERE_API_KEY="your-key"  # or use Ollama for local-only

python examples/01_quickstart.py
```

## Examples

| File | Description | Key Features |
|------|-------------|--------------|
| [`01_quickstart.py`](01_quickstart.py) | Basic SDK usage | `fitz()` → `ingest()` → `ask()` |
| [`02_tabular_sql.py`](02_tabular_sql.py) | CSV → SQL queries | Native PostgreSQL tables, computed answers |
| [`03_local_ollama.py`](03_local_ollama.py) | 100% local setup | No API keys, Ollama + embedded PostgreSQL |
| [`04_multi_collection.py`](04_multi_collection.py) | Multiple knowledge bases | Isolated collections, domain separation |
| [`05_advanced_queries.py`](05_advanced_queries.py) | Query intelligence | Keyword matching, comparisons, aggregations |

## Example 1: Quickstart (90% of use cases)

```python
from fitz_ai import fitz

f = fitz(collection="my_docs")
f.ingest("./docs")
answer = f.ask("What is the refund policy?")

print(answer.text)
for source in answer.provenance:
    print(f"  - {source.source_id}")
```

## Example 2: Tabular Data

```python
from fitz_ai import fitz

f = fitz(collection="sales")
f.ingest("./data/sales.csv")

# Natural language → SQL → computed answer
answer = f.ask("What is the total revenue by region?")
```

## Example 3: Local-Only (No API Keys)

```bash
# Install and start Ollama
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

```python
from fitz_ai import fitz

# Fitz auto-detects Ollama when no API keys are set
f = fitz(collection="private_docs")
f.ingest("./confidential")
answer = f.ask("Summarize the key points")
# Everything runs locally - no data leaves your machine
```

## Example 4: Multiple Collections

```python
from fitz_ai import fitz

# Separate knowledge bases
engineering = fitz(collection="engineering")
engineering.ingest("./eng_docs")

hr = fitz(collection="hr")
hr.ingest("./hr_docs")

# Query the right collection
eng_answer = engineering.ask("What's our tech stack?")
hr_answer = hr.ask("What's the PTO policy?")
```

## Example 5: Advanced Query Features

```python
from fitz_ai import fitz

f = fitz(collection="bugs")
f.ingest("./bug_reports")

# Exact keyword matching - only returns BUG-1001, not similar IDs
answer = f.ask("What is BUG-1001?")

# Comparison queries - retrieves both entities
answer = f.ask("Compare Pro vs Enterprise plan")

# Aggregation queries - uses hierarchical summaries
answer = f.ask("What are the main trends this quarter?")

# Honest responses when info isn't available
answer = f.ask("What is BUG-9999?")  # "I cannot find BUG-9999..."
```

## Running the Examples

Each example is self-contained and creates temporary test data:

```bash
# Basic SDK usage
python examples/01_quickstart.py

# Tabular/SQL queries (creates sample CSV)
python examples/02_tabular_sql.py

# Local Ollama setup (requires Ollama running)
python examples/03_local_ollama.py

# Multiple collections
python examples/04_multi_collection.py

# Advanced query patterns
python examples/05_advanced_queries.py
```

## CLI Quick Reference

```bash
# One-liner RAG
fitz quickstart ./docs "What is this about?"

# Step-by-step
fitz init                    # Interactive setup
fitz ingest ./docs           # Ingest documents
fitz query "Your question"   # Query
fitz chat                    # Interactive chat

# Management
fitz collections             # Manage collections
fitz keywords list           # View extracted keywords
fitz doctor                  # System diagnostics
```

## Configuration

Fitz works out of the box, but you can customize via `~/.fitz/config/fitz_rag.yaml`:

```yaml
# LLM provider
chat:
  plugin_name: cohere  # or openai, anthropic, local_ollama
  kwargs:
    models:
      smart: command-a-03-2025
      fast: command-r7b-12-2024

# Embeddings
embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

# Storage (always PostgreSQL)
vector_db: pgvector
vector_db_kwargs:
  mode: local  # embedded PostgreSQL, zero config
  # mode: external
  # connection_string: postgresql://user:pass@host:5432/db
```

## More Resources

- [Full Documentation](../docs/)
- [Configuration Guide](../docs/CONFIG.md)
- [CLI Reference](../docs/CLI.md)
- [Architecture](../docs/ARCHITECTURE.md)
