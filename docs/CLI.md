<!-- docs/CLI.md -->
# Fitz AI - CLI Documentation

## Overview

Fitz provides a clean, minimal command-line interface for local-first RAG. Version 0.10.0 consolidates the workflow into a single `fitz query` command that handles both document registration and querying.

---

## Quick Start

```bash
# Zero-config RAG in one command
fitz query "What are the main topics?" --source ./docs

# Or step by step
fitz init                                        # Setup wizard
fitz query "What's in my docs?" --source ./docs  # Register docs + query
fitz query "Follow-up question"                  # Query existing collection
```

---

## Commands

### `fitz query`

The main entry point. Point at documents and ask questions. Combines document registration and querying into one command.

```bash
fitz query "Your question"
fitz query "What is this about?" --source ./docs
fitz query "Summarize the key points" -c my_collection --source ./docs
fitz query --chat                                # Interactive multi-turn mode
fitz query --chat -c my_collection               # Chat with specific collection
```

**Arguments:**
- `QUESTION` - Question to ask (optional when using `--chat`)

**Options:**
- `-s, --source PATH` - Path to documents (file or directory). Registers documents before querying.
- `-c, --collection NAME` - Collection name
- `-e, --engine NAME` - Engine to use
- `--chat` - Interactive multi-turn chat mode

**How `--source` works:**
When you pass `--source`, Fitz registers (ingests) the documents into the collection, then runs your query against them. On subsequent queries without `--source`, Fitz uses the already-registered collection.

**Chat mode:**
- Type questions naturally
- Exit with `exit`, `quit`, or Ctrl+C

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

### `fitz config`

View or edit configuration, and run system diagnostics.

```bash
fitz config              # Show config summary
fitz config --raw        # Show raw YAML
fitz config --json       # Output as JSON
fitz config --path       # Show config file path
fitz config --edit       # Open in editor
fitz config --doctor     # Run system diagnostics
fitz config --test       # Test actual LLM connections
```

**Options:**
- `-p, --path` - Show config file path
- `--json` - Output as JSON
- `--raw` - Show raw YAML
- `-e, --edit` - Open config in editor
- `-d, --doctor` - Run system diagnostics (Python version, dependencies, PostgreSQL status, API keys)
- `-t, --test` - Test actual LLM connections

---

### `fitz serve`

Start the REST API server.

```bash
fitz serve
fitz serve --host 0.0.0.0 -p 8080
fitz serve --reload              # Auto-reload for development
```

**Options:**
- `-h, --host` - Host to bind to (default: 127.0.0.1)
- `-p, --port` - Port to bind to (default: 8000)
- `--reload` - Enable auto-reload

**API Endpoints:**
- `POST /query` - Query the knowledge base
- `POST /chat` - Multi-turn conversation
- `POST /point` - Register (index) documents
- `GET /collections` - List collections
- `GET /collections/{name}` - Collection details
- `DELETE /collections/{name}` - Delete a collection
- `GET /health` - Health check

---

### `fitz reset`

Reset the pgserver database. Use when pgserver hangs or gets corrupted.

```bash
fitz reset
fitz reset --force       # Skip confirmation prompt
```

**Options:**
- `-f, --force` - Skip confirmation prompt

---

### `fitz eval`

Evaluation and benchmarking tools.

```bash
fitz eval governance-stats    # Show governance decision statistics
fitz eval beir                # Run BEIR retrieval benchmark
fitz eval rgb                 # Run RGB robustness tests
fitz eval fitz-gov            # Run fitz-gov governance benchmark
fitz eval dashboard           # Display benchmark results dashboard
fitz eval all                 # Run all benchmarks
```

---

## Configuration

The `fitz init` command creates `.fitz/config/fitz_krag.yaml` in your project root:

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

# Retrieval strategy
retrieval:
  plugin_name: dense
  collection: default
  top_k: 5

# Rerank provider (presence enables reranking automatically)
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
fitz query "What's in my docs?" --source ./docs
```

### Multi-Turn Exploration

```bash
# Register docs and enter chat mode
fitz query --chat --source ./docs -c my_project

# Or chat with an existing collection
fitz query --chat -c my_project
```

### Development Workflow

```bash
fitz config --doctor --test      # Verify setup and test connections
fitz query "Summarize the architecture" --source ./project -c project
fitz query --chat -c project     # Interactive exploration
```

---

## Getting Help

```bash
fitz --help
fitz <command> --help
fitz eval --help
```
