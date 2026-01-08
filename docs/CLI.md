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
fitz ingest ./docs -e graphrag           # Override engine for this command
```

**Options:**
- `SOURCE` - Path to documents (file or directory)
- `-c, --collection` - Collection name
- `-e, --engine` - Engine to use (overrides default)
- `-y, --yes` - Non-interactive mode
- `-f, --force` - Force re-ingest all files
- `-H, --hierarchy` - Enable hierarchical summaries (L0/L1/L2)
- `-a, --artifacts` - Artifacts to generate (e.g., "all", "architecture_narrative")

**Hierarchical Mode (`-H`):**
Generates multi-level summaries during ingestion:
- L0: Original chunks (for specific queries)
- L1: Document/group summaries
- L2: Corpus summary (for "what are the trends?" queries)

---

### `fitz query`

Query the knowledge base. Uses the default engine (set via `fitz engine`).

```bash
fitz query "Your question"
fitz query "What is RAG?" -c my_collection
fitz query "Question" -e clara           # Override engine for this query
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
2. Detected keywords are stored in `.fitz/keywords/{collection}.yaml`
3. At query time, keywords in your question pre-filter chunks
4. Semantic search runs only on chunks containing the keyword

This ensures queries like "What happened with TC-1001?" only return chunks mentioning TC-1001, not similar IDs like TC-1002.

---

### `fitz map`

Visualize knowledge base as an interactive graph.

```bash
fitz map
fitz map -c my_collection
fitz map -o output.html --no-open
fitz map -t 0.9                     # Higher similarity threshold
fitz map --no-similarity-edges      # Hide similarity connections
```

**Options:**
- `-c, --collection` - Collection name
- `-o, --output` - Output file path
- `--no-open` - Don't open in browser
- `--rebuild` - Force rebuild embeddings
- `-t, --similarity-threshold` - Minimum similarity for edges (default: 0.8)
- `--no-similarity-edges` - Don't show similarity edges

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
- Available services (Ollama, Qdrant, FAISS)
- API key configuration

---

### `fitz engine`

View or set the default engine for all commands.

```bash
fitz engine              # Interactive engine selection
fitz engine --list       # List available engines
fitz engine fitz_rag     # Set default to fitz_rag
fitz engine clara        # Set default to clara
fitz engine graphrag     # Set default to graphrag
```

**Options:**
- `NAME` - Engine name to set as default
- `-l, --list` - List available engines

All other commands (`ingest`, `query`, `chat`, etc.) will use the default engine.
Override for a single command with `--engine`:

```bash
fitz query "question" -e clara   # Use clara just for this query
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

vector_db:
  plugin_name: qdrant
  kwargs:
    host: "localhost"
    port: 6333

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
# Start Ollama
ollama serve
ollama pull llama3.2

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Initialize and use
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
fitz map -c project          # Visualize knowledge
```

---

## Getting Help

```bash
fitz --help
fitz <command> --help
```
