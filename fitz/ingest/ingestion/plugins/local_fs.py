from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable

from fitz.ingest.ingestion.base import RawDocument

logger = logging.getLogger(__name__)


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
            except Exception as e:
                # Log file read failures but continue processing
                logger.warning(f"Skipping {path}: {type(e).__name__}: {e}")
                continue

            yield RawDocument(
                path=str(path),
                content=content,
                metadata={
                    "source": "local_fs",
                    **(kwargs.get("metadata") or {}),
                },
            )
