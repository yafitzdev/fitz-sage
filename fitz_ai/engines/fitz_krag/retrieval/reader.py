# fitz_ai/engines/fitz_krag/retrieval/reader.py
"""
Content reader — reads raw file content for addresses, extracts line ranges.

Addresses are lightweight pointers; reading fetches the actual content.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
    from fitz_ai.engines.fitz_krag.ingestion.section_store import SectionStore

logger = logging.getLogger(__name__)


class ContentReader:
    """Reads raw file content for addresses, extracts line ranges."""

    def __init__(
        self,
        raw_store: "RawFileStore",
        section_store: "SectionStore | None" = None,
        config: "FitzKragConfig | None" = None,
    ):
        self._raw_store = raw_store
        self._section_store = section_store
        self._config = config

    def read(self, addresses: list[Address], limit: int) -> list[ReadResult]:
        """Read content for top addresses."""
        results: list[ReadResult] = []
        for addr in addresses[:limit]:
            result = self._read_address(addr)
            if result:
                results.append(result)
        return results

    def _read_address(self, addr: Address) -> ReadResult | None:
        """Read content for a single address."""
        if addr.kind == AddressKind.SYMBOL:
            return self._read_symbol(addr)
        elif addr.kind == AddressKind.FILE:
            return self._read_file(addr)
        elif addr.kind == AddressKind.CHUNK:
            return self._read_chunk(addr)
        elif addr.kind == AddressKind.SECTION:
            return self._read_section(addr)
        return None

    def _read_symbol(self, addr: Address) -> ReadResult | None:
        """Read symbol content from raw file by line range."""
        raw_file = self._raw_store.get(addr.source_id)
        if not raw_file:
            logger.debug(f"Raw file not found for symbol address: {addr.source_id}")
            return None

        lines = raw_file["content"].splitlines()
        start = addr.metadata.get("start_line", 1) - 1  # 0-indexed
        end = addr.metadata.get("end_line", len(lines))
        code = "\n".join(lines[max(0, start) : end])

        return ReadResult(
            address=addr,
            content=code,
            file_path=raw_file["path"],
            line_range=(start + 1, end),
        )

    def _read_file(self, addr: Address) -> ReadResult | None:
        """Read entire file content."""
        raw_file = self._raw_store.get(addr.source_id)
        if not raw_file:
            return None

        return ReadResult(
            address=addr,
            content=raw_file["content"],
            file_path=raw_file["path"],
        )

    def _read_chunk(self, addr: Address) -> ReadResult | None:
        """Read chunk content (stored in address metadata)."""
        text = addr.metadata.get("text", "")
        if not text:
            return None

        return ReadResult(
            address=addr,
            content=text,
            file_path=addr.location,
        )

    def _read_section(self, addr: Address) -> ReadResult | None:
        """Read section content from section store."""
        if not self._section_store:
            return None

        section_id = addr.metadata.get("section_id")
        if not section_id:
            return None

        section = self._section_store.get(section_id)
        if not section:
            return None

        raw_file = self._raw_store.get(addr.source_id)
        file_path = raw_file["path"] if raw_file else "unknown"

        content = section["content"]
        metadata: dict[str, Any] = {
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
            "section_title": section["title"],
            "section_level": section["level"],
        }

        # Add breadcrumb and child TOC when section context is enabled
        if self._config and self._config.include_section_context:
            breadcrumb = self._build_breadcrumb(section)
            if breadcrumb:
                content = f"[{breadcrumb}]\n{content}"
                metadata["breadcrumb"] = breadcrumb

            children = self._section_store.get_children(section_id)
            if children:
                child_titles = "\n".join(f"  - {c['title']}" for c in children)
                content = f"{content}\n\nSubsections:\n{child_titles}"
                metadata["child_count"] = len(children)

        return ReadResult(
            address=addr,
            content=content,
            file_path=file_path,
            metadata=metadata,
        )

    def _build_breadcrumb(self, section: dict[str, Any]) -> str:
        """Walk up parent_section_id chain to build a breadcrumb path.

        Caps at 5 levels to prevent runaway chains.
        """
        if not self._section_store:
            return ""

        titles: list[str] = []
        parent_id = section.get("parent_section_id")
        depth = 0

        while parent_id and depth < 5:
            parent = self._section_store.get(parent_id)
            if not parent:
                break
            titles.append(parent["title"])
            parent_id = parent.get("parent_section_id")
            depth += 1

        if not titles:
            return ""

        titles.reverse()
        return " > ".join(titles)
