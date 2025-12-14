from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from fitz.ingest.ingestion.base import RawDocument


class LocalFSIngestPlugin:
    plugin_name = "local"

    def __init__(self, **_: Any) -> None:
        pass

    def ingest(self, source: str, kwargs: Dict[str, Any]) -> Iterable[RawDocument]:
        base = Path(source)

        paths = [base] if base.is_file() else list(base.glob("**/*"))

        for path in paths:
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
