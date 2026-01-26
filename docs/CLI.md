# Fitz AI - CLI Documentation

## Overview

Fitz provides a clean, minimal command-line interface for local-first RAG.

---

## Quick Start

```bash
# Zero-config RAG in one command
fitz quickstart ./docs "What are the main topics?"

# Or step by step
fitz init                    # Setup wizard
fitz ingest ./docs           # Ingest documents
fitz query "Your question"   # Query knowledge base
```

---

## Commands

### `fitz quickstart`

One-command RAG: ingest docs and ask a question.

```bash
fitz quickstart [SOURCE] [QUESTION]
fitz quickstart ./docs "What is this about?"
fitz quickstart ./docs "Summarize the key points" -c my_collection
```

**Options:**
- `SOURCE` - Path to documents (file or directory)
- `QUESTION` - Question to ask
- `-c, --collection` - Collection name (default: "quickstart")
- `-e, --engine` - Engine to use
- `-v, --verbose` - Show detailed progress

---

### `fitz init`

Interactive setup wizard that detects your system and creates configuration.

```bash
fitz init              # Interactive wizard
fitz init -y           # Auto-detect and use defaults
fitz init --show       # Preview config without saving
```

**Options:**
- `-y, --non-interactive` - Use defaults without prompting
- `-s, --show` - Preview config without saving

---

### `fitz ingest`

Ingest documents into the knowledge base. Uses the default engine (set via `fitz engine`).

```bash
fitz ingest [SOURCE]
fitz ingest ./docs -c my_collection
fitz ingest ./docs -y                    # Non-interactive
fitz ingest ./docs -H                    # With hierarchical summaries
fitz ingest ./docs --artifacts all       # Generate all artifacts
fitz ingest ./docs -f                    # Force re-ingest all files
fitz ingest ./docs -e my_engine          # Override engine for this command
```

**Options:**
- `SOURCE` - Path to documents (file or directory)
- `-c, --collection` - Collection name
- `-e, --engine` - Engine to use (overrides default)
- `-y, --yes` - Non-interactive mode
- `-f, --force` - Force re-ingest all files
- `-a, --artifacts` - Artifacts to generate (e.g., "all", "architecture_narrative")

**Hierarchical Summaries (always on):**
Every ingestion automatically generates multi-level summaries:
- L0: Original chunks (for specific queries)
- L1: Document/group summaries
- L2: Corpus summary (for "what are the trends?" queries)

---

### `fitz query`

Query the knowledge base. Uses the default engine (set via `fitz engine`).

```bash
fitz query "Your question"
fitz query "What is RAG?" -c my_collection
fitz query "Question" -e my_engine       # Override engine for this query
```

**Options:**
- `QUESTION` - Your query text
- `-c, --collection` - Collection name
- `-e, --engine` - Engine to use (overrides default)

---

### `fitz chat`

Interactive multi-turn conversation with your knowledge base.

```bash
fitz chat
fitz chat -c my_collection
```

**Options:**
- `-c, --collection` - Collection to chat with
- `-e, --engine` - Engine to use

**In chat:**
- Type questions naturally
- Exit with `exit`, `quit`, or Ctrl+C

---

### `fitz collections`

Manage collections (list, info, delete).

```bash
fitz collections
```

Interactive menu to:
- List all collections
- View collection info
- Delete collections

---

### `fitz keywords`

Manage keyword vocabulary for exact matching. Keywords are auto-detected during ingestion and used to pre-filter chunks before semantic search.

```bash
fitz keywords list                      # List all keywords
fitz keywords list -c my_collection     # For specific collection
fitz keywords add "CUSTOM-ID"           # Add custom keyword
fitz keywords add "MyTerm" --category custom  # With category
fitz keywords remove "TC-1001"          # Remove a keyword
fitz keywords clear                     # Clear all keywords
```

**Options:**
- `-c, --collection` - Collection name (uses default if not specified)
- `--category` - Category for new keywords (e.g., testcase, ticket, custom)

**Auto-detected patterns:**
- Test cases: TC-1001, testcase_42
- Tickets: JIRA-4521, BUG-789
- Versions: v2.0.1, 1.0.0-beta
- Pull requests: PR #123, PR-456
- People: John Smith (when mentioned multiple times)
- Files: config.yaml, report.pdf

**How it works:**
1. During ingestion, Fitz scans chunks for identifier patterns
2. Detected keywords are stored in PostgreSQL (`keywords` table per collection)
3. At query time, keywords in your question pre-filter chunks
4. Semantic search runs only on chunks containing the keyword

This ensures queries like "What happened with TC-1001?" only return chunks mentioning TC-1001, not similar IDs like TC-1002.

---

### `fitz serve`

Start the REST API server.

```bash
fitz serve
fitz serve -h 0.0.0.0 -p 8080
fitz serve --reload              # Auto-reload for development
```

**Options:**
- `-h, --host` - Host to bind to (default: 127.0.0.1)
- `-p, --port` - Port to bind to (default: 8000)
- `--reload` - Enable auto-reload

**API Endpoints:**
- `POST /query` - Query the knowledge base
- `POST /ingest` - Ingest documents
- `GET /collections` - List collections
- `GET /health` - Health check

---

### `fitz config`

View or edit configuration.

```bash
fitz config              # Show config summary
fitz config --raw        # Show raw YAML
fitz config --json       # Output as JSON
fitz config --path       # Show config file path
fitz config --edit       # Open in editor
```

**Options:**
- `-p, --path` - Show config file path
- `--json` - Output as JSON
- `--raw` - Show raw YAML
- `-e, --edit` - Open config in editor

---

### `fitz doctor`

System diagnostics and health check.

```bash
fitz doctor              # Quick check
fitz doctor -v           # Verbose output
fitz doctor --test       # Test actual connections
```

**Options:**
- `-v, --verbose` - Show detailed output
- `-t, --test` - Run connectivity tests

**Checks:**
- Python version
- Configuration files
- Required dependencies
- PostgreSQL/pgserver status
- Available services (Ollama, pgvector)
- API key configuration

---

### `fitz engine`

View or set the default engine for all commands.

```bash
fitz engine              # Interactive engine selection
fitz engine --list       # List available engines
fitz engine fitz_rag     # Set default to fitz_rag
```

**Options:**
- `NAME` - Engine name to set as default
- `-l, --list` - List available engines

All other commands (`ingest`, `query`, `chat`, etc.) will use the default engine.
Override for a single command with `--engine`:

```bash
fitz query "question" -e my_engine   # Use custom engine for this query
```

---

## Configuration

The `fitz init` command creates `.fitz/config/fitz_rag.yaml` in your project root:

```yaml
chat:
  plugin_name: cohere
  kwargs:
    models:
      smart: command-a-03-2025
      fast: command-r7b-12-2024
    temperature: 0.2

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
  collection: default
  top_k: 5

# Rerank provider (used only if retrieval uses reranking)
rerank:
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5

# Parser choice controls VLM usage
chunking:
  default:
    parser: docling_vision  # or "docling" for no VLM
    plugin_name: recursive
```

---

## Environment Variables

```bash
# Cohere (recommended)
export COHERE_API_KEY="your-key"

# OpenAI (alternative)
export OPENAI_API_KEY="your-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
```

---

## Common Workflows

### Local-First Setup

```bash
# Start Ollama (for local LLM)
ollama serve
ollama pull llama3.2

# Initialize and use (PostgreSQL starts automatically via pgserver)
fitz init
fitz ingest ./docs -y
fitz query "What's in my docs?"
```

### With Hierarchical Summaries

```bash
# Ingest with hierarchy for analytical queries
fitz ingest ./docs -H -c my_kb

# Ask trend/summary questions
fitz query "What are the main themes?" -c my_kb
```

### Development Workflow

```bash
fitz doctor --test           # Verify setup
fitz ingest ./project -H     # Ingest with summaries
fitz chat -c project         # Interactive exploration
```

---

## Getting Help

```bash
fitz --help
fitz <command> --help
```
