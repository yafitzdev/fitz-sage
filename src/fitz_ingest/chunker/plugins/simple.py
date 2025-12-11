from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SimpleChunker:
    """
    Very simple text chunker used for tests and default ingestion.

    - Splits text into fixed-size character blocks.
    - Strips whitespace.
    - Emits dict-based "chunks" with:
        {
            "text": <chunk_text>,
            "metadata": { ... base_meta ... }
        }
    """
    plugin_name: str = "base"
    chunk_size: int = 1000

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split the given text into fixed-size pieces and attach base metadata.

        Parameters
        ----------
        text:
            Full document text.
        base_meta:
            Base metadata dict (e.g. {"source_file": "<path>"}).

        Returns
        -------
        List[Dict[str, Any]]:
            List of normalized chunk dicts.
        """
        chunks: List[Dict[str, Any]] = []
        size = int(self.chunk_size)
        length = len(text)

        for i in range(0, length, size):
            piece = text[i: i + size].strip()
            if not piece:
                continue

            chunks.append(
                {
                    "text": piece,
                    # ensure each chunk has its own metadata copy
                    "metadata": dict(base_meta),
                }
            )

        return chunks
