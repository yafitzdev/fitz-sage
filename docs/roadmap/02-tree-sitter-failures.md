# docs/roadmap/02-tree-sitter-failures.md
# Problem: Non-Python code silently loses all symbols

## Status: Open
## Impact: High
## Effort: Low

## Problem

TypeScript, JavaScript, Java, and Go symbol extraction depends on tree-sitter,
a native C library that must be compiled or installed separately. When tree-sitter
fails to import (common on Windows, fresh installs, or constrained environments),
the error is silently logged at DEBUG level. The user gets **zero symbol extraction**
with no warning — their entire codebase is indexed as flat text.

Python extraction uses stdlib `ast` and always works. This creates a quality gap
where Python projects get excellent retrieval and non-Python projects get mediocre
retrieval with no indication of why.

## Evidence

From `engines/fitz_krag/progressive/worker.py` lines ~186-216:

```python
# Tree-sitter strategies are lazy-initialized
try:
    from fitz_ai.engines.fitz_krag.ingestion.strategies.typescript import (
        TypeScriptIngestStrategy,
    )
    strategy = TypeScriptIngestStrategy()
except ImportError:
    logger.debug("tree-sitter not available for TypeScript")  # Silent!
    return IngestResult()  # Empty — no symbols, no warning
```

The user sees no error. `fitz query` works but retrieval quality is degraded
because code search falls back to generic chunk matching instead of symbol-level
search.

## What Goes Wrong

1. User points at a TypeScript project: `fitz point ./my-ts-app`
2. Registration succeeds (files listed in manifest)
3. Symbol extraction silently returns empty for all `.ts`/`.tsx` files
4. User queries "where is the login handler?" — gets poor results
5. User has no idea tree-sitter is missing or that quality is degraded

## Proposed Fix

### Immediate (Low effort)

1. **Upgrade log level from DEBUG to WARNING** on first tree-sitter failure per language
2. **Show CLI notice** during `fitz point` if any language strategies fail:
   ```
   Warning: tree-sitter not installed. TypeScript/Java/Go symbol extraction
   disabled. Install with: pip install tree-sitter tree-sitter-typescript
   Code will still be indexed as text but without function/class-level search.
   ```
3. **Add `progress` callback** to report extraction status per language

### Follow-up (Medium effort)

4. **Regex fallback** for basic symbol extraction without tree-sitter:
   - TypeScript/JS: `/(?:export\s+)?(?:function|class|const|interface)\s+(\w+)/`
   - Java: `/(?:public|private|protected)?\s*(?:class|interface|enum)\s+(\w+)/`
   - Go: `/func\s+(?:\([^)]+\)\s+)?(\w+)/`

   This captures ~80% of symbols without any native dependency. Not as accurate
   as AST parsing but far better than nothing.

5. **Bundle pre-built tree-sitter wheels** or add as optional dependency group:
   ```toml
   [project.optional-dependencies]
   code = ["tree-sitter>=0.20", "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
   ```

## Affected Files

| File | Change |
|------|--------|
| `engines/fitz_krag/progressive/worker.py` | Upgrade log level, add progress callback |
| `engines/fitz_krag/ingestion/strategies/typescript.py` | Add regex fallback |
| `engines/fitz_krag/ingestion/strategies/java.py` | Add regex fallback |
| `engines/fitz_krag/ingestion/strategies/go.py` | Add regex fallback |
| `pyproject.toml` | Add `[code]` optional dependency group |

## Acceptance Criteria

- [ ] User sees WARNING (not debug) when tree-sitter unavailable
- [ ] CLI shows notice during `fitz point` for affected languages
- [ ] Regex fallback extracts function/class names without tree-sitter
- [ ] Python extraction unaffected (still uses stdlib ast)
