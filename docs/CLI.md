<!-- docs/CLI.md -->
# Fitz AI - CLI Documentation

## Overview

Fitz provides a clean, minimal command-line interface for local-first RAG. Version 0.10.0 consolidates the workflow into a single `fitz query` command that handles both document registration and querying.

---

## Quick Start

```bash
# Zero-config RAG in one command
fitz query "What are the main topics?" --source ./docs

# Config is auto-created on first run at .fitz/config.yaml
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
- `POST /query` - Query the knowledge base (optional `source` field to register documents)
- `POST /chat` - Multi-turn conversation
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

## Configuration

Config is auto-created on first run at `.fitz/config.yaml` in your project root:

```yaml
chat_fast: cohere/command-r7b-12-2024
chat_balanced: cohere/command-r-08-2024
chat_smart: cohere/command-a-03-2025
embedding: cohere/embed-v4.0
rerank: cohere/rerank-v3.5      # or null to disable
vision: null                     # or cohere (for docling_vision parser)
collection: default
parser: glm_ocr                  # or docling, docling_vision

vector_db: pgvector
vector_db_kwargs:
  mode: local  # or "external" with connection_string
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

# Query (PostgreSQL starts automatically via pgserver, config auto-created on first run)
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
# Edit .fitz/config.yaml to configure providers
fitz query "Summarize the architecture" --source ./project -c project
fitz query --chat -c project     # Interactive exploration
```

---

## Getting Help

```bash
fitz --help
fitz <command> --help
```
