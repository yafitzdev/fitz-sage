# Ingestion Pipeline

How documents flow through Fitz from files to searchable chunks.

---

## Overview

The ingestion pipeline transforms your documents into searchable knowledge:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Files  │ →  │  Parse  │ →  │  Chunk  │ →  │  Embed  │ →  │  Store  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                   │              │              │              │
              ParsedDoc       Chunks[]      Vectors[]     VectorDB
```

**Key features:**
- **Incremental** - Only processes new/changed files
- **Format-aware** - PDFs, code, markdown handled differently
- **Enrichable** - Optional LLM enhancement (summaries, hierarchies)

---

## Pipeline Stages

### 1. Source (File Discovery)

Finds files to ingest and provides local access.

```
Source.discover(path) → [SourceFile, SourceFile, ...]
```

| Component | Purpose |
|-----------|---------|
| `FileSystemSource` | Local filesystem discovery |
| `SourceFile` | Abstraction for file access (URI, local path, metadata) |

**What happens:**
1. Recursively scans the input path
2. Filters by supported extensions
3. Returns `SourceFile` objects with metadata

---

### 2. Diff (Change Detection)

Determines which files need processing.

```
Differ.diff(files, state) → (to_ingest, to_skip, to_delete)
```

| Component | Purpose |
|-----------|---------|
| `FileScanner` | Computes file hashes |
| `Differ` | Compares against stored state |
| `IngestStateManager` | Persists file hashes and chunker IDs |

**Change detection:**

| Change Type | Detection Method | Action |
|-------------|------------------|--------|
| New file | Not in state | Ingest |
| Modified file | Content hash changed | Re-ingest |
| Config changed | Chunker ID changed | Re-chunk |
| Deleted file | In state, not on disk | Mark deleted |
| Unchanged | Hash + chunker match | Skip |

**State file:** `.fitz/ingest_state.json`

```json
{
  "files": {
    "/path/to/doc.md": {
      "content_hash": "abc123...",
      "chunker_id": "recursive:1000:200",
      "chunk_ids": ["chunk1", "chunk2"],
      "ingested_at": "2024-01-15T10:30:00"
    }
  }
}
```

---

### 3. Parse (Document Extraction)

Converts files to structured documents.

```
Parser.parse(SourceFile) → ParsedDocument
```

| Component | Purpose |
|-----------|---------|
| `ParserRouter` | Routes files to parsers by extension |
| `DoclingParser` | PDFs, DOCX, images via Docling |
| `DoclingVisionParser` | Same + VLM for figure descriptions |
| `PlainTextParser` | Text files, markdown, code |

**Parsed document structure:**

```python
ParsedDocument(
    source_uri="file:///path/to/doc.pdf",
    elements=[
        TextElement(text="Chapter 1", level=1),
        TextElement(text="Introduction paragraph..."),
        TableElement(data=[...]),
        ImageElement(description="[Figure]"),  # or VLM description
    ],
    metadata={"title": "Document Title", ...}
)
```

**Parser selection:**

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.pdf`, `.docx`, `.pptx` | Docling | Structure extraction |
| `.png`, `.jpg` | Docling | OCR + optional VLM |
| `.md`, `.txt`, `.py` | PlainText | Direct text reading |

**VLM for figures:** Set `chunking.default.parser: docling_vision` to enable AI-generated figure descriptions instead of `[Figure]` placeholders.

---

### 4. Chunk (Text Splitting)

Splits documents into retrieval-sized pieces.

```
Chunker.chunk(ParsedDocument) → [Chunk, Chunk, ...]
```

| Component | Purpose |
|-----------|---------|
| `ChunkingRouter` | Routes by extension or uses default |
| `RecursiveChunker` | General-purpose, respects structure |
| `MarkdownChunker` | Header-aware splitting |
| `PythonCodeChunker` | AST-aware, keeps functions intact |

**Chunk structure:**

```python
Chunk(
    id="abc123...",           # Deterministic hash
    content="The text...",    # Chunk content
    metadata={
        "source_file": "/path/to/doc.pdf",
        "chunk_index": 0,
        "page": 1,
        "heading": "Chapter 1",
    }
)
```

**Chunker ID:**

Each chunker has a unique ID encoding its parameters:

```
recursive:1000:200  # plugin:chunk_size:overlap
python_code:1500:100
markdown:800:150
```

If the chunker ID changes (e.g., you change chunk_size), affected files are automatically re-chunked.

---

### 5. Embed (Vectorization)

Converts text to vectors for similarity search.

```
Embedder.embed_batch(texts) → [vector, vector, ...]
```

| Component | Purpose |
|-----------|---------|
| `Embedder` (YAML plugin) | Text-to-vector conversion |

**Batch processing:**
- Chunks are embedded in batches for efficiency
- Typical batch size: 96 chunks
- Embedding model determines vector dimension (e.g., 1024 for Cohere)

---

### 6. Store (Vector Database)

Persists vectors for retrieval.

```
VectorDB.upsert(collection, points)
```

| Component | Purpose |
|-----------|---------|
| `VectorDBWriter` | Writes to configured database |

**Point structure:**

```python
{
    "id": "chunk_abc123",
    "vector": [0.1, 0.2, ...],  # 1024 dims
    "payload": {
        "content": "The text...",
        "source_file": "/path/to/doc.pdf",
        "chunk_index": 0,
        ...
    }
}
```

---

## Enrichment Pipeline (Optional)

The enrichment pipeline adds LLM-generated enhancements.

```
┌─────────────────────────────────────────────────────────────────┐
│  Enrichment Pipeline (optional, runs after chunking)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Summaries  │  │  Entities   │  │  Hierarchy              │  │
│  │             │  │             │  │                         │  │
│  │ Per-chunk   │  │ Extract:    │  │ Level 0: Chunks         │  │
│  │ LLM summary │  │ - classes   │  │ Level 1: Group summaries│  │
│  │             │  │ - functions │  │ Level 2: Corpus summary │  │
│  │             │  │ - concepts  │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Summaries

LLM-generated descriptions for better search.

```yaml
enrichment:
  summary:
    enabled: true  # Warning: 1 LLM call per chunk!
```

**Use case:** When chunk content is dense code or technical text, summaries provide natural language hooks for retrieval.

### Entities

Extracts named entities and concepts.

```yaml
enrichment:
  entities:
    enabled: true
    types: [class, function, api, person, organization]
```

**Result:** Entities stored in `chunk.metadata["entities"]`

### Hierarchy

Multi-level summaries for analytical queries.

```yaml
enrichment:
  hierarchy:
    enabled: true
    group_by: source_file  # Or use semantic clustering
```

**Levels:**
- **Level 0:** Original chunks (unchanged)
- **Level 1:** Group summaries (e.g., per-file)
- **Level 2:** Corpus summary (all groups)

**Use case:** "What are the main themes?" retrieves L1/L2 summaries instead of random chunks.

---

## Incremental Ingestion

Fitz only processes what's changed:

```bash
$ fitz ingest ./docs

Scanning... 847 files
  → 12 new files
  → 3 modified files
  → 832 unchanged (skipped)

Ingesting 15 files...
```

### What triggers re-ingestion?

| Change | Re-ingest? | Why |
|--------|------------|-----|
| File content changed | Yes | Content hash differs |
| Chunk size changed | Yes | Chunker ID differs |
| Embedding model changed | No | Vectors regenerated |
| New file added | Yes | Not in state |
| File deleted | Mark deleted | Clean up vectors |

### Force re-ingestion

```bash
fitz ingest ./docs --force  # Re-ingest everything
```

---

## File Format Support

### Documents

| Format | Parser | Features |
|--------|--------|----------|
| PDF | Docling | Tables, figures, sections |
| DOCX | Docling | Styles, tables |
| PPTX | Docling | Slides as sections |
| HTML | Docling | Structure extraction |

### Code

| Format | Chunker | Features |
|--------|---------|----------|
| Python | `python_code` | AST-aware, preserves functions |
| Markdown | `markdown` | Header-aware splitting |
| Other code | `recursive` | Respects indentation |

### Text

| Format | Parser | Notes |
|--------|--------|-------|
| `.txt` | PlainText | Direct reading |
| `.md` | PlainText | Preserves formatting |
| `.json`, `.yaml` | PlainText | Treated as text |

---

## CLI Commands

```bash
# Basic ingestion
fitz ingest ./docs

# Specify collection
fitz ingest ./docs --collection my_project

# Force re-ingest
fitz ingest ./docs --force

# Enable hierarchy summaries
fitz ingest ./docs --hierarchy

# Non-interactive mode
fitz ingest ./docs -y
```

---

## Python API

```python
import fitz_ai

# Simple ingestion
fitz_ai.ingest("./docs")

# With options
fitz_ai.ingest(
    "./docs",
    collection="my_project",
    force=True,
)
```

### Advanced usage

```python
from fitz_ai.ingestion.diff.executor import DiffIngestExecutor
from fitz_ai.ingestion.parser import ParserRouter
from fitz_ai.ingestion.chunking.router import ChunkingRouter
from fitz_ai.ingestion.state import IngestStateManager

# Build components
parser_router = ParserRouter(docling_parser="docling_vision")
chunking_router = ChunkingRouter.from_config(config)
state_manager = IngestStateManager()

# Create executor
executor = DiffIngestExecutor(
    state_manager=state_manager,
    vector_db_writer=vector_db,
    embedder=embedder,
    parser_router=parser_router,
    chunking_router=chunking_router,
    collection="my_collection",
    embedding_id="cohere:embed-english-v3.0",
)

# Run ingestion
summary = executor.run(path="./docs", force=False)
print(f"Ingested {summary.ingested} files")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_ai/ingestion/diff/executor.py` | Main orchestrator |
| `fitz_ai/ingestion/parser/router.py` | Parser selection |
| `fitz_ai/ingestion/chunking/router.py` | Chunker selection |
| `fitz_ai/ingestion/state/manager.py` | State persistence |
| `fitz_ai/ingestion/enrichment/pipeline.py` | Enrichment orchestrator |
| `fitz_ai/cli/commands/ingest.py` | CLI command |

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) - VLM and rerank control
- [PLUGINS.md](PLUGINS.md) - Creating custom chunkers/parsers
