# tools/contract_map/

Static analysis tool that enforces the architectural layer dependency rules defined in `CLAUDE.md`.

Detects forbidden cross-layer imports at the module level so violations are caught in CI before they reach runtime.

## Rules enforced

- `core/` — no imports from `engines/`, `ingestion/`, `retrieval/`, `llm/`, `vector_db/`
- `retrieval/`, `llm/`, `ingestion/` — may only import from `core/`
- `engines/` — may import `core/`, `llm/`, `vector_db/`, `storage/`, `retrieval/`
- `vector_db/` — may import `core/`, `storage/`
- `runtime/`, `cli/` — unrestricted

## Usage

```bash
python -m tools.contract_map              # Print violations
python -m tools.contract_map --fail-on-errors   # Exit non-zero on any violation (CI mode)
```
