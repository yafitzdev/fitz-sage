# fitz_ai/engines/fitz_krag/retrieval/strategies/llm_code_search.py
"""
LLM-based structural code search strategy.

Sends a compact AST manifest to the LLM and lets it reason about which files
are relevant. Falls back to hybrid CodeSearchStrategy on any failure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from fitz_ai.engines.fitz_krag.types import Address, AddressKind

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
    from fitz_ai.engines.fitz_krag.ingestion.symbol_store import SymbolStore
    from fitz_ai.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
    from fitz_ai.llm.factory import ChatFactory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are selecting files from a codebase to answer a question.\n\n"
    "Question: {query}\n\n"
    "Below is the structural index of every indexed file. Each entry shows "
    "file path, classes, functions, and imports.\n\n"
    "{structural_index}\n\n"
    "Select 5-10 relevant files for answering the question. Include files that:\n"
    "- Contain the code being asked about\n"
    "- Define protocols, base classes, or types used by relevant code\n"
    "- Contain configuration or factory patterns that affect the relevant code\n"
    "- Are in the same directory or package as the most relevant files\n\n"
    "Err on the side of including MORE files — missing a relevant file is worse "
    "than including an extra one.\n\n"
    "Return ONLY a JSON array of file paths:\n"
    '```json\n["path/to/file1.py", "path/to/file2.py", ...]\n```'
)


class ManifestBuilder:
    """Builds a compact structural manifest from the symbol index."""

    def __init__(
        self,
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        max_chars: int = 120_000,
    ):
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._max_chars = max_chars
        self._cached_manifest: str | None = None
        self._cached_file_data: dict[str, dict] | None = None
        self._cached_symbol_count: int | None = None

    def build(self) -> tuple[str, dict[str, dict]]:
        """Build manifest text and file data index.

        Returns:
            (manifest_text, file_data_by_path) where file_data_by_path maps
            path -> {"raw_file_id": str, "symbols": list[dict]}
        """
        manifest_data = self._symbol_store.get_structural_manifest()
        symbol_count = sum(len(f["symbols"]) for f in manifest_data)

        if (
            self._cached_manifest is not None
            and self._cached_symbol_count == symbol_count
        ):
            return self._cached_manifest, self._cached_file_data  # type: ignore[return-value]

        file_data: dict[str, dict] = {}
        for f in manifest_data:
            file_data[f["path"]] = {
                "raw_file_id": f["raw_file_id"],
                "symbols": f["symbols"],
            }

        # Build full manifest lines per file
        lines_by_path: dict[str, str] = {}
        for f in manifest_data:
            lines_by_path[f["path"]] = self._format_file(f)

        manifest_text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))

        # Smart truncation if over budget
        if len(manifest_text) > self._max_chars:
            manifest_text = self._truncate(manifest_data, lines_by_path)

        self._cached_manifest = manifest_text
        self._cached_file_data = file_data
        self._cached_symbol_count = symbol_count
        return manifest_text, file_data

    def _format_file(self, file_entry: dict) -> str:
        """Format a single file's symbols into compact manifest text."""
        path = file_entry["path"]
        symbols = file_entry["symbols"]
        parts = [f"## {path}"]

        classes = []
        functions = []
        imports_set: set[str] = set()

        for sym in symbols:
            kind = sym["kind"]
            name = sym["name"]
            sig = sym.get("signature") or ""

            if kind == "class":
                # Collect methods for this class
                methods = [
                    s
                    for s in symbols
                    if s["kind"] == "method"
                    and s["qualified_name"].startswith(sym["qualified_name"] + ".")
                ]
                if methods:
                    method_strs = []
                    for m in methods:
                        m_sig = m.get("signature") or ""
                        ret = self._extract_return_type(m_sig)
                        method_strs.append(f"{m['name']} -> {ret}")
                    classes.append(f"{name}[{', '.join(method_strs)}]")
                else:
                    classes.append(name)

            elif kind == "function":
                ret = self._extract_return_type(sig)
                params = self._extract_params(sig)
                functions.append(f"{name}({params}) -> {ret}")

            # Collect imports
            for imp in sym.get("imports", []):
                imports_set.add(imp)

        if classes:
            parts.append(f"classes: {', '.join(classes)}")
        if functions:
            parts.append(f"functions: {', '.join(functions)}")
        if imports_set:
            parts.append(f"imports: {', '.join(sorted(imports_set))}")

        return "\n".join(parts)

    def _truncate(self, manifest_data: list[dict], lines_by_path: dict[str, str]) -> str:
        """Smart truncation: 3-pass, least-connected files lose detail first."""
        reverse_counts = self._import_store.get_reverse_counts()

        # Sort files by connectivity (least connected first)
        sorted_files = sorted(
            manifest_data,
            key=lambda f: reverse_counts.get(f["raw_file_id"], 0),
        )

        # Pass 1: strip imports from least-connected files
        for f in sorted_files:
            path = f["path"]
            lines = lines_by_path[path].split("\n")
            lines_by_path[path] = "\n".join(
                ln for ln in lines if not ln.startswith("imports:")
            )
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        # Pass 2: strip functions from least-connected files
        for f in sorted_files:
            path = f["path"]
            lines = lines_by_path[path].split("\n")
            lines_by_path[path] = "\n".join(
                ln
                for ln in lines
                if not ln.startswith("functions:") and not ln.startswith("imports:")
            )
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        # Pass 3: path-only for least-connected files
        for f in sorted_files:
            path = f["path"]
            lines_by_path[path] = f"## {path}"
            text = "\n".join(lines_by_path[p] for p in sorted(lines_by_path))
            if len(text) <= self._max_chars:
                return text

        return "\n".join(lines_by_path[p] for p in sorted(lines_by_path))

    @staticmethod
    def _extract_return_type(sig: str) -> str:
        if " -> " in sig:
            return sig.split(" -> ", 1)[1].strip()
        return "None"

    @staticmethod
    def _extract_params(sig: str) -> str:
        if "(" in sig and ")" in sig:
            inner = sig[sig.index("(") + 1 : sig.rindex(")")]
            # Strip 'self, ' prefix for methods
            if inner.startswith("self, "):
                inner = inner[6:]
            elif inner == "self":
                inner = ""
            return inner.strip()
        return ""


class LlmCodeSearchStrategy:
    """LLM-based structural code search with hybrid fallback.

    Sends a compact AST manifest to the LLM, which selects relevant files.
    On any failure, transparently falls back to hybrid CodeSearchStrategy.
    """

    def __init__(
        self,
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        chat_factory: "ChatFactory",
        config: "FitzKragConfig",
        fallback_strategy: "CodeSearchStrategy",
    ):
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._chat_factory = chat_factory
        self._config = config
        self._fallback = fallback_strategy
        self._manifest_builder = ManifestBuilder(symbol_store, import_store)

        # Forwarded attributes — engine wires these after construction
        self._hyde_generator: Any = None
        self._raw_store: Any = None

    def retrieve(
        self,
        query: str,
        limit: int,
        detection: Any = None,
        *,
        query_vector: list[float] | None = None,
        hyde_vectors: list[list[float]] | None = None,
    ) -> list[Address]:
        """Retrieve code addresses via LLM structural search, with fallback."""
        try:
            return self._llm_retrieve(query, limit)
        except Exception as e:
            logger.info(f"LLM code search failed, falling back to hybrid: {e}")
            return self._fallback.retrieve(
                query,
                limit,
                detection=detection,
                query_vector=query_vector,
                hyde_vectors=hyde_vectors,
            )

    def _llm_retrieve(self, query: str, limit: int) -> list[Address]:
        """Core LLM-based retrieval path."""
        manifest_text, file_data = self._manifest_builder.build()
        if not manifest_text.strip():
            raise ValueError("Empty manifest — no indexed symbols")

        selected_paths = self._llm_select_files(manifest_text, query)
        if not selected_paths:
            raise ValueError("LLM returned no file selections")

        # Map paths to file IDs
        path_to_id = {path: data["raw_file_id"] for path, data in file_data.items()}
        selected_ids = [path_to_id[p] for p in selected_paths if p in path_to_id]

        if not selected_ids:
            raise ValueError("No selected paths matched indexed files")

        # Expand imports (depth 1, forward only)
        expanded_ids = self._expand_imports(selected_ids)
        all_ids = list(dict.fromkeys(selected_ids + expanded_ids))

        # Convert to Address objects
        return self._files_to_addresses(all_ids, file_data, limit)

    def _llm_select_files(self, manifest: str, query: str) -> list[str]:
        """Ask the LLM which files are relevant."""
        prompt = _SYSTEM_PROMPT.format(query=query, structural_index=manifest)
        chat = self._chat_factory("fast")
        response = chat.chat([{"role": "user", "content": prompt}])

        # Parse JSON array from response
        text = response.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            raise ValueError(f"No JSON array in LLM response: {text[:200]}")

        parsed = json.loads(text[start:end])
        if not isinstance(parsed, list):
            raise ValueError(f"LLM returned non-list: {type(parsed)}")

        return [str(p) for p in parsed if isinstance(p, str) and p.strip()]

    def _expand_imports(self, file_ids: list[str]) -> list[str]:
        """Forward-only depth-1 import expansion."""
        expanded: list[str] = []
        seen = set(file_ids)
        for fid in file_ids:
            imports = self._import_store.get_imports(fid)
            for edge in imports:
                target = edge.get("target_file_id")
                if target and target not in seen:
                    expanded.append(target)
                    seen.add(target)
        return expanded

    def _files_to_addresses(
        self, file_ids: list[str], file_data: dict[str, dict], limit: int
    ) -> list[Address]:
        """Convert file IDs to symbol Address objects.

        Scores decrease by file position (first file = most relevant) so that
        diverse files survive ranking. Caps symbols per file to prevent one
        large file from monopolizing all read slots.
        """
        # Build reverse mapping: file_id -> path
        id_to_path = {data["raw_file_id"]: path for path, data in file_data.items()}

        max_symbols_per_file = max(3, limit // max(1, len(file_ids)))
        addresses: list[Address] = []
        for file_rank, fid in enumerate(file_ids):
            path = id_to_path.get(fid)
            if not path or path not in file_data:
                continue
            # Score decreases with file rank: 0.80, 0.77, 0.74, ...
            # Moderate base — LLM file selection is a fast heuristic, not a
            # precision signal, so scores must stay competitive with agentic
            # results rather than dominating them.
            file_score = max(0.5, 0.80 - file_rank * 0.03)
            file_symbols = file_data[path]["symbols"][:max_symbols_per_file]
            for sym_rank, sym in enumerate(file_symbols):
                # Within a file, minor decrease so class > methods in ranking
                score = file_score - sym_rank * 0.001
                addresses.append(
                    Address(
                        kind=AddressKind.SYMBOL,
                        source_id=fid,
                        location=sym["qualified_name"],
                        summary=f"{sym['kind']} {sym['name']}",
                        score=score,
                        metadata={
                            "symbol_id": "",
                            "name": sym["name"],
                            "qualified_name": sym["qualified_name"],
                            "kind": sym["kind"],
                            "start_line": sym["start_line"],
                            "end_line": sym["end_line"],
                            "signature": sym.get("signature"),
                        },
                    )
                )
        return addresses[:limit]
