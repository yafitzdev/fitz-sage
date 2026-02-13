# docs/roadmap/01-docx-pptx-chunking.md
# Problem: DOCX/PPTX have no structure-aware chunking

## Status: Done (heading extraction + cache; section extraction was already working)
## Impact: High
## Effort: Medium

## Problem

DOCX and PPTX files are parsed through Docling (which extracts text correctly) but
then chunked by `RecursiveChunker` — a generic paragraph/sentence splitter with no
awareness of document structure. A 50-page Word document gets split into arbitrary
~500-char chunks with no heading hierarchy, no section boundaries, and no slide
separation for presentations.

This is the same class of problem PDF had before `PdfSectionChunker` was added.
The retrieval quality for DOCX/PPTX is significantly worse than for Markdown or PDF
because chunks lose their structural context.

## Evidence

Current chunker routing (from `ingestion/chunking/router.py`):

| Extension | Chunker | Structure-aware? |
|-----------|---------|-----------------|
| `.py` | PythonCodeChunker | Yes (AST) |
| `.md` | MarkdownChunker | Yes (headings) |
| `.pdf` | PdfSectionChunker | Yes (heuristic sections) |
| `.csv` | TableChunker | Yes (schema) |
| `.docx` | RecursiveChunker | **No** |
| `.pptx` | RecursiveChunker | **No** |
| `.html` | RecursiveChunker | **No** |

## What Goes Wrong

1. **DOCX**: Heading hierarchy (H1/H2/H3) is discarded. A section titled
   "3.2 Authentication Flow" becomes an anonymous text blob. Cross-references
   like "as described in Section 3.2" can't resolve.

2. **PPTX**: Slide boundaries are lost. Content from slide 5 merges with slide 6.
   Speaker notes (if extracted) mix with slide text. Title slides lose their
   structural role.

3. **Both**: Table of contents, bullet hierarchies, and numbered lists lose nesting.

## Proposed Fix

### Option A: Docling-aware chunker (Recommended)

Docling already extracts document structure (headings, tables, lists, pages).
Instead of converting to flat text then re-chunking, use Docling's structured
output directly:

```python
# ingestion/chunking/plugins/docling_sections.py
class DoclingChunker:
    """Structure-aware chunker using Docling's document model."""
    supported_extensions = {".docx", ".pptx", ".html"}

    def chunk(self, parsed_doc: ParsedDocument) -> list[Chunk]:
        # Use Docling's heading hierarchy to create sections
        # Each section = one chunk with metadata:
        #   - section_title, section_level, page_number
        # Split oversized sections at paragraph boundaries
```

This requires changing the parser → chunker interface to pass structured output
(not just flat text) for rich documents.

### Option B: Post-hoc heading detection (Simpler)

Similar to `PdfSectionChunker` — detect headings from the flat text using
formatting heuristics. Less accurate but doesn't require interface changes.

### Option C: PPTX-specific slide chunker

For PPTX specifically, parse with python-pptx directly (skip Docling) and
create one chunk per slide with title as section heading.

## Affected Files

| File | Change |
|------|--------|
| `ingestion/chunking/plugins/` | New chunker plugin(s) |
| `ingestion/chunking/router.py` | Register new chunkers for `.docx`, `.pptx` |
| `ingestion/parser/router.py` | Possibly expose structured output from Docling |
| `engines/fitz_krag/ingestion/section_store.py` | Store heading metadata |

## Acceptance Criteria

- [ ] DOCX heading hierarchy preserved in chunk metadata
- [ ] PPTX slides produce separate chunks with slide title
- [ ] Section-level retrieval quality matches Markdown for equivalent content
- [ ] Existing RecursiveChunker still used as fallback for unknown formats
