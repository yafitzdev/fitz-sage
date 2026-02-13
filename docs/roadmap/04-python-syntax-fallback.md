# docs/roadmap/04-python-syntax-fallback.md
# Problem: Python syntax errors = zero extraction

## Status: Done (regex fallback in PythonCodeIngestStrategy)
## Impact: Medium
## Effort: Low

## Problem

The Python code ingest strategy uses stdlib `ast.parse()` which requires valid
syntax. A single syntax error anywhere in a `.py` file means the entire file gets
`IngestResult()` — zero symbols, zero references, zero imports. The file becomes
invisible to code search and only findable through generic chunk matching.

This affects real-world codebases with WIP branches, generated code with template
placeholders, Python 2/3 compatibility files, or partial files saved mid-edit.

## Evidence

From `engines/fitz_krag/ingestion/strategies/python_code.py`:

```python
def extract(self, content: str, metadata: dict) -> IngestResult:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return IngestResult()  # Everything lost
```

The `PythonCodeChunker` has the same issue — syntax errors fall back to treating
the entire file as a single chunk:

```python
# chunking/plugins/python_code.py
try:
    tree = ast.parse(content)
except SyntaxError:
    return [Chunk(content=content, metadata=metadata)]  # One giant chunk
```

## What Goes Wrong

1. User has a Python project with one file containing a syntax error (e.g., missing
   closing parenthesis, f-string with backslash in Python <3.12)
2. `ast.parse()` raises `SyntaxError`
3. Symbol extraction returns empty — no functions, classes, or constants indexed
4. Code chunker creates one chunk with the entire file content
5. User asks "where is the handle_auth function?" — symbol search misses it entirely
6. Generic chunk search might find it but with poor ranking (no symbol metadata)

## Proposed Fix

### Regex fallback (Low effort, high ROI)

When `ast.parse()` fails, fall back to regex-based extraction that catches ~90%
of symbols:

```python
import re

def _regex_fallback(content: str) -> IngestResult:
    """Extract symbols via regex when AST parsing fails."""
    symbols = []

    # Functions
    for m in re.finditer(r'^(?:async\s+)?def\s+(\w+)\s*\(', content, re.MULTILINE):
        symbols.append(Symbol(name=m.group(1), kind="function", start_line=...))

    # Classes
    for m in re.finditer(r'^class\s+(\w+)\s*[:\(]', content, re.MULTILINE):
        symbols.append(Symbol(name=m.group(1), kind="class", start_line=...))

    # Constants
    for m in re.finditer(r'^([A-Z][A-Z_0-9]+)\s*=', content, re.MULTILINE):
        symbols.append(Symbol(name=m.group(1), kind="constant", start_line=...))

    return IngestResult(symbols=symbols)
```

This won't capture signatures, type hints, or references — but it finds the
symbol names and their locations, which is what code search needs most.

### Line-level chunking fallback

For the chunker, instead of one giant chunk, split at function/class boundaries
detected by regex:

```python
except SyntaxError:
    # Split at def/class lines instead of one blob
    return _split_at_definitions(content, metadata)
```

## Affected Files

| File | Change |
|------|--------|
| `engines/fitz_krag/ingestion/strategies/python_code.py` | Add `_regex_fallback()` |
| `ingestion/chunking/plugins/python_code.py` | Add `_split_at_definitions()` |

## Acceptance Criteria

- [ ] Files with syntax errors still produce symbol entries (name + kind + line)
- [ ] Chunker splits at function/class boundaries even without valid AST
- [ ] Valid Python files unaffected (still use full AST parsing)
- [ ] Log WARNING when falling back to regex (user knows quality is degraded)
