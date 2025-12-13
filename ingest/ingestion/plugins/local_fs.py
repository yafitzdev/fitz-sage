# ingest/ingestion/plugins/local_fs.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from ingest.ingestion.base import RawDocument


class LocalFSIngestPlugin:
    plugin_name = "local"

    def __init__(self, **_: Any) -> None:
        pass

    def ingest(self, source: str, kwargs: Dict[str, Any]) -> Iterable[RawDocument]:
        base = Path(source)

        for path in base.glob("**/*"):
            if not path.is_file():
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            yield RawDocument(
                path=str(path),
                content=content,
                metadata={
                    "source": "local_fs",
                    **(kwargs.get("metadata") or {}),
                },
            )
