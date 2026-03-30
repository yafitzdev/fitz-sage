# Code Symbol Extraction

## Problem

Traditional RAG chunks code into fixed-size text blocks. A 512-token window doesn't know where a function ends or a class begins. You get half a method in one chunk and the other half in the next — broken syntax, lost context, no way to answer "what calls this function?"

```python
# Traditional RAG: chunk 1 (ends mid-function)
def authenticate(user):
    if not user.token:

# Traditional RAG: chunk 2 (starts mid-function)
        raise AuthError()
    return validate(user.token)
```

KRAG doesn't chunk code. It extracts **symbols** — addressable units with qualified names, signatures, references, and import relationships.

## How It Works

KRAG uses language-specific strategies to parse source files into `SymbolEntry` objects via AST parsing. Each symbol is a complete, meaningful unit — never split.

### What Gets Extracted

| Symbol Kind | Example | What's Captured |
|------------|---------|-----------------|
| **function** | `def authenticate(user)` | Name, qualified name, signature, references, source |
| **class** | `class AuthService` | Name, base classes, source, methods extracted separately |
| **method** | `AuthService.validate_token` | Qualified as `module.Class.method`, signature, references |
| **constant** | `MAX_RETRIES = 3` | UPPER_CASE module/class-level assignments |
| **interface** | `interface UserProps` | TypeScript/Java interfaces |
| **type** | `type Config struct` | Go structs, TypeScript type aliases |

Each symbol carries:

```python
@dataclass
class SymbolEntry:
    name: str              # "authenticate"
    qualified_name: str    # "auth.service.AuthService.authenticate"
    kind: str              # "method"
    start_line: int        # 12
    end_line: int          # 18
    signature: str         # "def authenticate(self, user: User) -> bool"
    source: str            # Full source code of the symbol
    imports: list[str]     # What this file imports
    references: list[str]  # Symbols referenced in the body
```

### Language Support

| Language | Parser | Symbol Types |
|----------|--------|-------------|
| **Python** | `ast` (stdlib) | Functions, classes, methods, constants |
| **TypeScript/JavaScript** | tree-sitter | Functions, classes, methods, interfaces, type aliases, exports |
| **Go** | tree-sitter | Functions, methods (with receiver), structs, interfaces, const/var |
| **Java** | tree-sitter | Classes, interfaces, enums, records, methods, constructors, fields |

All strategies implement the same protocol:

```python
class IngestStrategy(Protocol):
    def content_types(self) -> set[str]: ...
    def extract(self, source: str, file_path: str) -> IngestResult: ...
```

Each returns an `IngestResult` containing symbols and import edges. The pipeline doesn't care which parser produced them.

### Fallback

If AST parsing fails (syntax errors, unsupported constructs), strategies fall back to regex extraction. Fewer symbols, less metadata, but the file still gets indexed.

## Import Graph

Symbol extraction also builds an **import graph** — file-level dependency edges stored in PostgreSQL:

```
auth/service.py ──imports──► auth/models.py
auth/service.py ──imports──► core/exceptions.py
api/routes.py   ──imports──► auth/service.py
```

This enables:
- **"What depends on AuthService?"** — reverse graph traversal, not text search
- **"What would break if I change this?"** — follow incoming edges
- **Context expansion** — when a symbol matches, pull in symbols from files it imports

Relative imports are resolved to absolute module names, so `from .models import User` in `auth/service.py` correctly links to `auth.models`.

## Storage

Symbols are stored in `krag_symbol_index` with:
- **Vector column** (`summary_vector`) — HNSW-indexed for semantic search
- **BM25 column** (`content_tsv`) — auto-generated tsvector over name + qualified_name + summary
- **Name indexes** — for direct name/qualified_name lookup
- **Keyword array** — enrichment-extracted keywords for exact matching
- **Entity array** — enrichment-extracted named entities

Three search paths hit the same table:
1. **Vector search** — semantic similarity on summary embeddings
2. **BM25 search** — full-text search on name + summary
3. **Name search** — ILIKE on name and qualified_name

The retrieval router fuses results from all three using RRF (Reciprocal Rank Fusion).

## Example

**Input:** `auth/service.py`

```python
import hashlib
from typing import Optional

class AuthService:
    """Handles user authentication."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def authenticate(self, user: User) -> bool:
        if not user.token:
            raise AuthError("Missing token")
        return self.validate_token(user.token)

    def validate_token(self, token: str) -> bool:
        hash_val = hashlib.sha256(token.encode()).hexdigest()
        return hash_val == self.secret_key
```

**Extracted symbols:**

| # | qualified_name | kind | signature | references |
|---|---------------|------|-----------|------------|
| 1 | `auth.service.AuthService` | class | `class AuthService` | — |
| 2 | `auth.service.AuthService.__init__` | method | `def __init__(self, secret_key: str)` | `self.secret_key` |
| 3 | `auth.service.AuthService.authenticate` | method | `def authenticate(self, user: User) -> bool` | `AuthError`, `self.validate_token` |
| 4 | `auth.service.AuthService.validate_token` | method | `def validate_token(self, token: str) -> bool` | `hashlib.sha256`, `self.secret_key` |

**Extracted imports:**

| target_module | import_names |
|--------------|-------------|
| `hashlib` | `hashlib` |
| `typing` | `Optional` |

Each symbol gets summarized by the enrichment pipeline, embedded, and stored in `krag_symbol_index`. At query time, "How does authentication work?" finds `AuthService.authenticate` via semantic search on its summary — not by hoping a text chunk contains the right keywords.

## Files

| Component | Path |
|-----------|------|
| Strategy protocol | `fitz_sage/engines/fitz_krag/ingestion/strategies/base.py` |
| Python strategy | `fitz_sage/engines/fitz_krag/ingestion/strategies/python_code.py` |
| TypeScript strategy | `fitz_sage/engines/fitz_krag/ingestion/strategies/typescript.py` |
| Go strategy | `fitz_sage/engines/fitz_krag/ingestion/strategies/go.py` |
| Java strategy | `fitz_sage/engines/fitz_krag/ingestion/strategies/java.py` |
| Symbol store | `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py` |
| Import graph store | `fitz_sage/engines/fitz_krag/ingestion/import_graph_store.py` |
| DB schema | `fitz_sage/engines/fitz_krag/ingestion/schema.py` |

## Standalone Code Retrieval

For use cases that don't need the full KRAG pipeline (no PostgreSQL, no pgvector, no docling), see the standalone `CodeRetriever` in `fitz_sage/code/`:

```bash
pip install fitz-sage[code]
```

```python
from fitz_sage.code import CodeRetriever

retriever = CodeRetriever(source_dir="./myproject", chat_factory=my_factory)
results = retriever.retrieve("How does auth work?")
```

Uses AST-based structural indexing and LLM file selection — same algorithm as KRAG's `LlmCodeSearchStrategy`, but reads from disk instead of PostgreSQL.

## Related Features

- [**KRAG**](../platform/krag.md) — The engine architecture that symbols plug into
- [**Hybrid Search**](../retrieval/hybrid-search.md) — BM25 + vector fusion used for symbol retrieval
- [**Keyword Vocabulary**](../retrieval/keyword-vocabulary.md) — Exact-match on function names, class names
- [**Entity Graph**](../retrieval/entity-graph.md) — Entity-based linking across symbols and sections
- [**Contextual Embeddings**](../retrieval/contextual-embeddings.md) — Summary-prefixed embeddings for disambiguation
