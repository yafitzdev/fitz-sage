# docs/roadmap/README.md
# Ingestion Quality Roadmap

Post-v0.9.0 problem areas in the ingestion pipeline, ordered by user impact.

PDF parsing is solved (pypdfium2 fast path + Docling OCR fallback with caching).
These are the remaining domains that need attention.

| # | Problem | File | Impact | Effort |
|---|---------|------|--------|--------|
| 1 | [DOCX/PPTX have no structure-aware chunking](./01-docx-pptx-chunking.md) | `ingestion/chunking/` | High | Medium |
| 2 | [Non-Python code silently loses all symbols](./02-tree-sitter-failures.md) | `progressive/worker.py` | High | Low |
| 3 | [CSV rows invisible to semantic search](./03-table-vector-gap.md) | `chunking/plugins/table.py` | Medium | Medium |
| 4 | [Python syntax errors = zero extraction](./04-python-syntax-fallback.md) | `strategies/python_code.py` | Medium | Low |

Each file is self-contained: problem, evidence, proposed fix, affected files.
