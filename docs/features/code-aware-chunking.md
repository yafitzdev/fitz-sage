# Code-Aware Chunking

## Problem

Naive chunking splits code mid-function, breaking syntax and losing context:

```python
# Chunk 1 (ends here)
def authenticate(user):
    if not user.token:

# Chunk 2 (starts here - broken!)
        raise AuthError()
    return validate(user.token)
```

A 50-line class becomes 3 fragments that don't make sense alone. Code needs **semantic boundaries** (functions, classes, methods), not arbitrary character limits.

## Solution: AST-Aware Chunking

Fitz uses Abstract Syntax Tree (AST) parsing to chunk code at logical boundaries:

```python
# Fitz chunking: entire function = 1 chunk
def authenticate(user):
    if not user.token:
        raise AuthError()
    return validate(user.token)
```

Functions, classes, and methods become individual searchable units with docstrings, imports, and type hints preserved.

## How It Works

### Language-Specific Strategies

| Language | Strategy | Boundaries |
|----------|----------|------------|
| **Python** | AST parsing via `ast` module | Classes, functions, methods, module docstrings |
| **JavaScript/TypeScript** | Tree-sitter parsing | Functions, classes, methods, exports |
| **Markdown** | Header-aware splits | H1/H2/H3 headers, code blocks kept intact |
| **PDF** | Section detection | Numbered sections (1.1, 2.3.1), keywords ("Abstract", "Conclusion") |
| **Other** | Sentence-based fallback | Sentence boundaries with configurable max size |

### Python Chunking Example

**Input:** `auth.py`

```python
"""Authentication module for user management."""

import hashlib
from typing import Optional

class AuthService:
    """Handles user authentication and token validation."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def authenticate(self, user: User) -> bool:
        """Authenticate user with token validation."""
        if not user.token:
            raise AuthError("Missing token")
        return self.validate_token(user.token)

    def validate_token(self, token: str) -> bool:
        """Validate token signature."""
        hash_val = hashlib.sha256(token.encode()).hexdigest()
        return hash_val == self.secret_key
```

**Output chunks:**

1. **Chunk 1 (module):**
   ```python
   """Authentication module for user management."""

   import hashlib
   from typing import Optional
   ```

2. **Chunk 2 (class + __init__):**
   ```python
   class AuthService:
       """Handles user authentication and token validation."""

       def __init__(self, secret_key: str):
           self.secret_key = secret_key
   ```

3. **Chunk 3 (authenticate method):**
   ```python
   def authenticate(self, user: User) -> bool:
       """Authenticate user with token validation."""
       if not user.token:
           raise AuthError("Missing token")
       return self.validate_token(user.token)
   ```

4. **Chunk 4 (validate_token method):**
   ```python
   def validate_token(self, token: str) -> bool:
       """Validate token signature."""
       hash_val = hashlib.sha256(token.encode()).hexdigest()
       return hash_val == self.secret_key
   ```

### Markdown Chunking Example

**Input:** `guide.md`

```markdown
# Installation Guide

Instructions for installing the system.

## Prerequisites

- Python 3.10+
- pip

## Installation Steps

1. Clone the repo
2. Run `pip install -e .`

### Advanced Options

Optional configuration flags:

\```bash
pip install -e ".[dev]"
\```
```

**Output chunks:**

1. **Chunk 1 (H1: Installation Guide):**
   ```markdown
   # Installation Guide

   Instructions for installing the system.
   ```

2. **Chunk 2 (H2: Prerequisites):**
   ```markdown
   ## Prerequisites

   - Python 3.10+
   - pip
   ```

3. **Chunk 3 (H2: Installation Steps + H3: Advanced Options):**
   ```markdown
   ## Installation Steps

   1. Clone the repo
   2. Run `pip install -e .`

   ### Advanced Options

   Optional configuration flags:

   \```bash
   pip install -e ".[dev]"
   \```
   ```

Note: Code blocks are kept intact, never split mid-block.

## Key Design Decisions

1. **Always-on** - Code-aware chunking is the default for recognized file types. No configuration needed.

2. **Fallback gracefully** - If AST parsing fails, fall back to sentence-based chunking.

3. **Preserve context** - Imports, docstrings, decorators, and type hints stay attached to their functions/classes.

4. **Respect max size** - Large functions/classes are split at method boundaries if they exceed max chunk size (default: 1000 tokens).

5. **Language detection** - File extension determines chunking strategy (.py → Python, .md → Markdown, etc.).

## Configuration

Minimal configuration required. Chunking strategy is determined by file type.

Override chunking strategy in `config.yaml`:

```yaml
chunking:
  default:
    plugin_name: semantic  # or "sentence", "fixed_size"
    kwargs:
      max_chunk_size: 1000  # tokens
      overlap: 200  # tokens
```

## Files

- **Python chunker:** `fitz_ai/ingestion/chunking/plugins/semantic.py` (AST-based)
- **Markdown chunker:** `fitz_ai/ingestion/chunking/plugins/markdown.py` (header-aware)
- **Sentence chunker:** `fitz_ai/ingestion/chunking/plugins/sentence.py` (fallback)
- **Chunking router:** `fitz_ai/ingestion/chunking/router.py`

## Benefits

| Naive Chunking | Code-Aware Chunking |
|----------------|---------------------|
| Splits mid-function | Entire functions as chunks |
| Broken syntax | Valid, runnable code blocks |
| Lost context | Docstrings + imports preserved |
| Random boundaries | Semantic boundaries (classes, methods) |

## Example Use Case: Codebase Search

**Query:** "How does the authentication module work?"

**Naive chunking:**
- Returns: 5 random fragments from auth.py
- Result: Broken code snippets, incomplete logic

**Code-aware chunking:**
- Returns:
  1. `AuthService` class definition + docstring
  2. `authenticate()` method (complete)
  3. `validate_token()` method (complete)
  4. Module docstring + imports
- Result: Complete, understandable code units

User can see:
- Full function signatures
- Complete logic flow
- Docstrings and type hints
- Import dependencies

## Supported File Types

| Extension | Chunking Strategy |
|-----------|------------------|
| `.py` | AST-aware (functions, classes, methods) |
| `.js`, `.ts`, `.jsx`, `.tsx` | Tree-sitter parsing (functions, classes) |
| `.md` | Header-aware (H1/H2/H3 boundaries) |
| `.pdf` | Section detection (numbered sections, keywords) |
| `.txt`, `.rst` | Sentence-based fallback |
| `.json`, `.yaml`, `.toml` | Fixed-size chunking (config files) |

## Dependencies

- **Python:** `ast` module (built-in)
- **JavaScript/TypeScript:** `tree-sitter` (optional, falls back to sentence-based if missing)
- **Markdown:** Pure Python implementation (no dependencies)
- **PDF:** Docling parser (optional, used if configured)

## Performance Considerations

- **Parsing overhead:** AST parsing adds ~10-50ms per file (negligible)
- **Memory:** AST trees held in memory during parsing (released after chunking)
- **Chunk count:** Code files produce fewer, larger chunks than naive chunking

## Related Features

- **Keyword Vocabulary** - Function names, class names automatically extracted and indexed
- **Hierarchical RAG** - File-level summaries for "What does this module do?" queries
- **Query Expansion** - Synonyms like "function" ↔ "method" ↔ "def" handled
