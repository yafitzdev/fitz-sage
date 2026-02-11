# fitz_ai/engines/fitz_krag/retrieval/expander.py
"""
Code expander — enriches read results with contextual code.

Adds imports, class context (signature + __init__), and same-file references
to provide the LLM with enough context to understand the code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
    from fitz_ai.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
    from fitz_ai.engines.fitz_krag.ingestion.symbol_store import SymbolStore

logger = logging.getLogger(__name__)


class CodeExpander:
    """Expands read results with contextual code."""

    def __init__(
        self,
        raw_store: "RawFileStore",
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        config: "FitzKragConfig",
    ):
        self._raw_store = raw_store
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._config = config

    def expand(self, read_results: list[ReadResult]) -> list[ReadResult]:
        """
        Expand read results with context.

        For SYMBOL addresses:
        1. Add file-level imports as a header
        2. If method, add class signature + __init__
        3. Add same-file referenced symbols
        """
        if self._config.max_expansion_depth == 0:
            return read_results

        expanded = list(read_results)

        for result in read_results:
            if result.address.kind != AddressKind.SYMBOL:
                continue

            # 1. File imports
            expanded = self._add_file_imports(expanded, result)

            # 2. Class context for methods
            if self._config.include_class_context:
                expanded = self._add_class_context(expanded, result)

        return self._deduplicate(expanded)

    def _add_file_imports(self, expanded: list[ReadResult], result: ReadResult) -> list[ReadResult]:
        """Add file-level import block as context."""
        raw_file = self._raw_store.get(result.address.source_id)
        if not raw_file:
            return expanded

        lines = raw_file["content"].splitlines()
        import_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                import_lines.append(line)
            elif stripped and not stripped.startswith("#") and import_lines:
                break  # Past import section

        if not import_lines:
            return expanded

        import_text = "\n".join(import_lines)

        # Check if we already have this import block
        for existing in expanded:
            if existing.file_path == result.file_path and existing.content == import_text:
                return expanded

        import_addr = Address(
            kind=AddressKind.FILE,
            source_id=result.address.source_id,
            location=f"{result.file_path} (imports)",
            summary="File imports",
            metadata={"context_type": "imports"},
        )
        expanded.append(
            ReadResult(
                address=import_addr,
                content=import_text,
                file_path=result.file_path,
                line_range=(1, len(import_lines)),
                metadata={"context_type": "imports"},
            )
        )
        return expanded

    def _add_class_context(
        self, expanded: list[ReadResult], result: ReadResult
    ) -> list[ReadResult]:
        """Add class signature + __init__ for method symbols."""
        kind = result.address.metadata.get("kind")
        if kind != "method":
            return expanded

        qualified = result.address.metadata.get("qualified_name", "")
        parts = qualified.rsplit(".", 2)
        if len(parts) < 3:
            return expanded

        # Find the class in symbol store
        class_qualified = ".".join(parts[:-1])
        # Search by name for the class
        class_name = parts[-2]
        class_results = self._symbol_store.search_by_name(class_name, limit=5)

        for cls in class_results:
            if cls["qualified_name"] == class_qualified and cls["kind"] == "class":
                # Read the class signature (first few lines)
                raw_file = self._raw_store.get(cls["raw_file_id"])
                if not raw_file:
                    break

                lines = raw_file["content"].splitlines()
                start = cls["start_line"] - 1
                # Just the class definition line + docstring
                class_header_end = min(start + 5, cls["end_line"])
                header = "\n".join(lines[start:class_header_end])

                # Check if already present
                already = any(
                    e.content == header and e.file_path == raw_file["path"] for e in expanded
                )
                if not already:
                    class_addr = Address(
                        kind=AddressKind.SYMBOL,
                        source_id=cls["raw_file_id"],
                        location=class_qualified,
                        summary=f"Class {class_name} (context)",
                        metadata={
                            "context_type": "class_header",
                            "start_line": cls["start_line"],
                            "end_line": class_header_end,
                        },
                    )
                    expanded.append(
                        ReadResult(
                            address=class_addr,
                            content=header,
                            file_path=raw_file["path"],
                            line_range=(cls["start_line"], class_header_end),
                            metadata={"context_type": "class_header"},
                        )
                    )
                break

        return expanded

    def _deduplicate(self, results: list[ReadResult]) -> list[ReadResult]:
        """Remove duplicate read results."""
        seen: set[tuple[str, str, int | None]] = set()
        deduped: list[ReadResult] = []
        for r in results:
            key = (
                r.file_path,
                r.address.location,
                r.line_range[0] if r.line_range else None,
            )
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped
