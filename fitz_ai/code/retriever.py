# fitz_ai/code/retriever.py
"""
Standalone code retrieval without database dependencies.

Builds structural index from files on disk, uses LLM to select relevant files,
expands imports and neighbors, reads and compresses content.

Usage:
    from fitz_ai.code import CodeRetriever
    from fitz_ai.llm.factory import get_chat_factory

    retriever = CodeRetriever(
        source_dir="./myproject",
        chat_factory=get_chat_factory("lmstudio"),
    )
    results = retriever.retrieve("How does authentication work?")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.code.indexer import build_file_list, build_import_graph, build_structural_index
from fitz_ai.code.prompts import EXPAND_AND_SELECT_PROMPT, HUB_FILES_HINT, NEIGHBOR_SCREEN_PROMPT
from fitz_ai.engines.fitz_krag.context.compressor import compress_python
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_ai.llm.factory import ChatFactory

logger = logging.getLogger(__name__)


class CodeRetriever:
    """Standalone code retrieval — no PostgreSQL, no pgvector, no docling.

    Builds a structural index from source files on disk, asks an LLM to select
    relevant files, expands via imports and directory neighbors, then reads
    and compresses the selected files.

    Args:
        source_dir: Root directory of the codebase.
        chat_factory: Factory returning ChatProvider instances per tier.
        llm_tier: Tier name passed to chat_factory for all LLM calls.
        max_manifest_chars: Max structural index size in characters.
        neighbor_screen_threshold: Max new sibling files before LLM screening.
        max_file_bytes: Max bytes to read per file for indexing.
        max_files: Max files to index.
        hub_import_threshold: Min forward imports to qualify as a hub file.
    """

    def __init__(
        self,
        source_dir: str | Path,
        chat_factory: "ChatFactory",
        *,
        llm_tier: str = "fast",
        max_manifest_chars: int = 120_000,
        neighbor_screen_threshold: int = 10,
        max_file_bytes: int = 50_000,
        max_files: int = 2000,
        hub_import_threshold: int = 5,
    ) -> None:
        self._source_dir = Path(source_dir).resolve()
        self._chat_factory = chat_factory
        self._llm_tier = llm_tier
        self._max_manifest_chars = max_manifest_chars
        self._neighbor_screen_threshold = neighbor_screen_threshold
        self._max_file_bytes = max_file_bytes
        self._max_files = max_files
        self._hub_import_threshold = hub_import_threshold

        # Lazy-built caches
        self._file_paths: list[str] | None = None
        self._structural_index: str | None = None
        self._import_graph: dict[str, set[str]] | None = None
        self._hub_files: list[tuple[str, int]] | None = None

    def retrieve(self, query: str, limit: int = 30) -> list[ReadResult]:
        """Full pipeline: index -> LLM select -> expand -> read -> compress.

        Args:
            query: Natural language question about the codebase.
            limit: Max files to return.

        Returns:
            List of ReadResult with compressed file content.
        """
        file_paths = self._get_file_paths()
        index_text = self._get_structural_index()
        import_graph = self._get_import_graph()

        if not index_text.strip():
            logger.warning("Empty structural index — no indexable files found")
            return []

        # LLM: combined expansion + file selection
        search_terms, selected_paths = self._llm_expand_and_select(index_text, query)
        if search_terms:
            logger.debug(f"Search terms: {search_terms}")

        # Validate selections against actual file list
        file_set = set(file_paths)
        selected = [p for p in selected_paths if p in file_set]
        if not selected:
            logger.warning("LLM returned no valid file selections")
            return []

        logger.info(f"LLM selected {len(selected)} files")

        # Import expansion (depth 1, forward only)
        import_expanded = self._expand_imports(selected, import_graph)
        logger.debug(f"Import expansion added {len(import_expanded)} files")

        # Facade expansion: __init__.py re-exports point to the actual
        # definitions. These are public API files and get priority over
        # regular depth-1 imports.
        facade_added: list[str] = []
        seen_d1 = set(selected) | set(import_expanded)
        for path in import_expanded:
            if path.endswith("__init__.py"):
                for dep in import_graph.get(path, set()):
                    if dep not in seen_d1:
                        facade_added.append(dep)
                        seen_d1.add(dep)
        if facade_added:
            logger.info(
                f"Facade expansion through __init__.py added {len(facade_added)} files"
            )

        # Hub file auto-inclusion — architectural hubs are always included
        seen_so_far = seen_d1
        hub_added: list[str] = []
        for path, _count in (self._hub_files or []):
            if path not in seen_so_far and path in file_set:
                hub_added.append(path)
                seen_so_far.add(path)
        if hub_added:
            logger.info(f"Hub auto-inclusion added {len(hub_added)} files")

        # Neighbor expansion (same-directory siblings)
        all_so_far = list(dict.fromkeys(
            selected + import_expanded + facade_added + hub_added
        ))
        neighbors = self._neighbor_expand(
            all_so_far, file_paths, import_graph, query
        )
        logger.debug(f"Neighbor expansion added {len(neighbors)} files")

        # Final file list — priority order:
        #   1. selected (LLM scan hits)
        #   2. hub (architectural orchestrators)
        #   3. facade (re-exported public API definitions)
        #   4. import (direct dependencies)
        #   5. neighbor (directory siblings)
        origin: dict[str, str] = {}
        for p in selected:
            origin[p] = "selected"
        for p in hub_added:
            if p not in origin:
                origin[p] = "hub"
        for p in facade_added:
            if p not in origin:
                origin[p] = "facade"
        for p in import_expanded:
            if p not in origin:
                origin[p] = "import"
        for p in neighbors:
            if p not in origin:
                origin[p] = "neighbor"

        all_files = list(dict.fromkeys(
            selected + hub_added + facade_added + import_expanded + neighbors
        ))[:limit]
        logger.info(
            f"Reading {len(all_files)} files "
            f"({len(selected)} selected, {len(hub_added)} hub, "
            f"{len(facade_added)} facade, {len(import_expanded)} import, "
            f"{len(neighbors)} neighbor)"
        )

        # Read and compress
        return self._read_and_compress(all_files, origin)

    def get_structural_index(self) -> str:
        """Return the structural index text (for inspection/debugging)."""
        return self._get_structural_index()

    def get_file_paths(self) -> list[str]:
        """Return the indexed file paths (triggers lazy build if needed)."""
        return self._get_file_paths()

    def get_hub_files(self) -> list[tuple[str, int]]:
        """Return hub files (triggers lazy build if needed).

        Returns:
            List of (path, import_count) sorted by import count descending.
        """
        self._get_structural_index()  # ensures _hub_files is populated
        return self._hub_files or []

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _get_file_paths(self) -> list[str]:
        if self._file_paths is None:
            self._file_paths = build_file_list(self._source_dir, self._max_files)
            logger.info(f"Indexed {len(self._file_paths)} files")
        return self._file_paths

    def _get_structural_index(self) -> str:
        if self._structural_index is None:
            file_paths = self._get_file_paths()
            import_graph = self._get_import_graph()
            # Connection counts for truncation priority
            conn_counts: dict[str, int] = {}
            for targets in import_graph.values():
                for t in targets:
                    conn_counts[t] = conn_counts.get(t, 0) + 1
            self._structural_index = build_structural_index(
                self._source_dir,
                file_paths,
                max_file_bytes=self._max_file_bytes,
                max_chars=self._max_manifest_chars,
                connection_counts=conn_counts,
            )
            # Compute hub files: files importing many other intra-project files
            hubs = [
                (path, len(targets))
                for path, targets in import_graph.items()
                if len(targets) > self._hub_import_threshold
            ]
            hubs.sort(key=lambda x: x[1], reverse=True)
            self._hub_files = hubs[:10]
            if self._hub_files:
                logger.info(
                    f"Found {len(hubs)} hub files "
                    f"(>{self._hub_import_threshold} imports), "
                    f"keeping top {len(self._hub_files)}"
                )
        return self._structural_index

    def _get_import_graph(self) -> dict[str, set[str]]:
        if self._import_graph is None:
            file_paths = self._get_file_paths()
            self._import_graph = build_import_graph(
                self._source_dir, file_paths, self._max_file_bytes
            )
        return self._import_graph

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    def _llm_expand_and_select(
        self, index: str, query: str
    ) -> tuple[list[str], list[str]]:
        """Combined query expansion + file selection in one LLM call."""
        hub_files = self._hub_files or []
        if hub_files:
            hub_lines = "\n".join(
                f"  - {path} (imports {count} subsystems)"
                for path, count in hub_files
            )
            hub_hint = HUB_FILES_HINT.format(hub_files=hub_lines)
        else:
            hub_hint = ""
        prompt = EXPAND_AND_SELECT_PROMPT.format(
            query=query, structural_index=index, hub_files_hint=hub_hint
        )
        chat = self._chat_factory(self._llm_tier)
        response = chat.chat([{"role": "user", "content": prompt}])
        text = response.strip()

        # Try combined JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}") + 1
        if brace_start >= 0 and brace_end > brace_start:
            try:
                parsed = json.loads(text[brace_start:brace_end])
                if isinstance(parsed, dict):
                    terms = [str(t) for t in parsed.get("search_terms", []) if isinstance(t, str)]
                    files = [str(f) for f in parsed.get("files", []) if isinstance(f, str)]
                    return terms, files
            except json.JSONDecodeError:
                pass

        # Fallback: plain JSON array
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                parsed_list = json.loads(text[start:end])
                if isinstance(parsed_list, list):
                    return [], [str(p) for p in parsed_list if isinstance(p, str)]
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse LLM response: {text[:200]}")
        return [], []

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand_imports(
        self, selected: list[str], import_graph: dict[str, set[str]]
    ) -> list[str]:
        """Forward-only depth-1 import expansion."""
        seen = set(selected)
        expanded: list[str] = []
        for path in selected:
            for dep in import_graph.get(path, set()):
                if dep not in seen:
                    expanded.append(dep)
                    seen.add(dep)
        return expanded

    def _neighbor_expand(
        self,
        selected: list[str],
        all_files: list[str],
        import_graph: dict[str, set[str]],
        query: str,
    ) -> list[str]:
        """Add sibling files from directories of selected files."""
        seen = set(selected)

        # Find triggered directories
        trigger_dirs: set[str] = set()
        for path in selected:
            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                trigger_dirs.add(parts[0])

        # Group all files by directory
        dir_files: dict[str, list[str]] = {}
        for path in all_files:
            parts = path.rsplit("/", 1)
            if len(parts) == 2:
                dir_files.setdefault(parts[0], []).append(path)

        neighbors: list[str] = []
        for d in trigger_dirs:
            new_siblings = [p for p in dir_files.get(d, []) if p not in seen]
            if not new_siblings:
                continue

            if len(new_siblings) > self._neighbor_screen_threshold:
                # LLM screening for large directories
                screened = self._screen_neighbors(
                    query, d, new_siblings, selected
                )
                new_siblings = screened

            for p in new_siblings:
                if p not in seen:
                    neighbors.append(p)
                    seen.add(p)

        return neighbors

    def _screen_neighbors(
        self,
        query: str,
        directory: str,
        siblings: list[str],
        triggers: list[str],
    ) -> list[str]:
        """LLM screens large directories for relevance."""
        # Build mini structural index for siblings
        sibling_index = build_structural_index(
            self._source_dir,
            siblings,
            max_file_bytes=self._max_file_bytes,
            max_chars=20_000,
        )
        trigger_file = next(
            (t for t in triggers if t.startswith(directory + "/")), directory
        )
        prompt = NEIGHBOR_SCREEN_PROMPT.format(
            query=query,
            trigger_file=trigger_file,
            sibling_index=sibling_index,
        )
        try:
            chat = self._chat_factory(self._llm_tier)
            response = chat.chat([{"role": "user", "content": prompt}])
            text = response.strip()
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    sibling_set = set(siblings)
                    return [str(p) for p in parsed if str(p) in sibling_set]
        except Exception as e:
            logger.debug(f"Neighbor screening failed for {directory}: {e}")

        # On failure, skip the large directory
        return []

    # ------------------------------------------------------------------
    # Read + compress
    # ------------------------------------------------------------------

    def _read_and_compress(
        self, file_paths: list[str], origin: dict[str, str]
    ) -> list[ReadResult]:
        """Read files from disk, compress Python, return ReadResults."""
        _ORIGIN_SCORES = {"selected": 1.0, "hub": 0.95, "facade": 0.92, "import": 0.9, "neighbor": 0.8}
        results: list[ReadResult] = []

        for path in file_paths:
            full_path = self._source_dir / path
            if not full_path.is_file():
                continue
            try:
                raw = full_path.read_bytes()
                content = raw.decode("utf-8", errors="replace")
            except OSError:
                continue

            # Compress Python files
            if path.endswith(".py"):
                content = compress_python(content)

            score = _ORIGIN_SCORES.get(origin.get(path, "neighbor"), 0.8)
            address = Address(
                kind=AddressKind.FILE,
                source_id=path,
                location=path,
                summary=path,
                score=score,
                metadata={"origin": origin.get(path, "neighbor")},
            )
            results.append(
                ReadResult(
                    address=address,
                    content=content,
                    file_path=path,
                )
            )

        return results
