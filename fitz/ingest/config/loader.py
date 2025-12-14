# ingest/config/loader.py
from __future__ import annotations

from pathlib import Path

import yaml

from ingest.config.schema import IngestConfig


class IngestConfigError(RuntimeError):
    pass


def load_ingest_config(path: str | Path) -> IngestConfig:
    p = Path(path)
    if not p.exists():
        raise IngestConfigError(f"Config file not found: {p}")

    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise IngestConfigError(f"Failed to load ingest config: {p}") from exc

    if not isinstance(data, dict):
        raise IngestConfigError("Ingest config root must be a mapping")

    try:
        return IngestConfig.model_validate(data)
    except Exception as exc:
        raise IngestConfigError("Invalid ingest configuration") from exc
