# tools/pre_release/

Pre-release verification steps run before tagging a new version.

| File | Purpose |
|---|---|
| `lint.py` | Runs Black, isort, and type-check passes |
| `prerelease.py` | Orchestrates the full pre-release checklist |

These are called by the `release-manager` agent and by `.github/workflows/release.yml`.

## Usage

```bash
python -m tools.pre_release   # via top-level entry point
```
