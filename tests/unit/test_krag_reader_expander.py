# tests/unit/test_krag_reader_expander.py
"""
Unit tests for ContentReader and CodeExpander.

ContentReader: reads raw file content for addresses, extracts line ranges.
CodeExpander: enriches read results with contextual code (imports, class headers).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_ai.engines.fitz_krag.retrieval.expander import CodeExpander
from fitz_ai.engines.fitz_krag.retrieval.reader import ContentReader
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_symbol_address(
    source_id: str = "file1",
    location: str = "mod.func",
    start_line: int = 4,
    end_line: int = 6,
    kind: str = "function",
    qualified_name: str = "mod.func",
    score: float = 0.9,
) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Symbol {location}",
        score=score,
        metadata={
            "start_line": start_line,
            "end_line": end_line,
            "kind": kind,
            "qualified_name": qualified_name,
        },
    )


def _make_file_address(
    source_id: str = "file1",
    location: str = "module.py",
) -> Address:
    return Address(
        kind=AddressKind.FILE,
        source_id=source_id,
        location=location,
        summary=f"File {location}",
    )


def _make_chunk_address(
    text: str = "chunk content here",
    location: str = "doc.md",
) -> Address:
    return Address(
        kind=AddressKind.CHUNK,
        source_id="chunk1",
        location=location,
        summary="A chunk",
        metadata={"text": text},
    )


def _make_section_address(
    section_id: str = "sec-001",
    source_id: str = "file1",
) -> Address:
    return Address(
        kind=AddressKind.SECTION,
        source_id=source_id,
        location="Section A",
        summary="Section about A",
        metadata={"section_id": section_id},
    )


RAW_FILE_CONTENT = (
    "import os\n"
    "import sys\n"
    "\n"
    "def func():\n"
    "    return 42\n"
    "\n"
    "def other():\n"
    "    pass\n"
)


def _make_raw_store(files: dict[str, dict] | None = None) -> MagicMock:
    """Create a mock RawFileStore.

    ``files`` maps source_id -> {"path": ..., "content": ...}.
    """
    store = MagicMock()
    if files is None:
        files = {
            "file1": {
                "path": "module.py",
                "content": RAW_FILE_CONTENT,
            }
        }
    store.get.side_effect = lambda sid: files.get(sid)
    return store


def _make_section_store(sections: dict[str, dict] | None = None) -> MagicMock:
    store = MagicMock()
    if sections is None:
        sections = {}
    store.get.side_effect = lambda sid: sections.get(sid)
    return store


def _make_config(
    max_expansion_depth: int = 1,
    include_class_context: bool = True,
) -> MagicMock:
    config = MagicMock()
    config.max_expansion_depth = max_expansion_depth
    config.include_class_context = include_class_context
    return config


# ===========================================================================
# TestContentReader
# ===========================================================================


class TestContentReader:
    """Tests for ContentReader."""

    def test_read_symbol(self):
        """Reads symbol content by extracting the correct line range."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        # Lines 4-5 of RAW_FILE_CONTENT: "def func():" and "    return 42"
        addr = _make_symbol_address(start_line=4, end_line=5, location="mod.func")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "def func():\n    return 42"
        assert r.file_path == "module.py"
        assert r.line_range == (4, 5)
        assert r.address is addr

    def test_read_symbol_missing_file(self):
        """When raw_store returns None the address is skipped."""
        raw_store = _make_raw_store(files={})  # empty -> always None
        reader = ContentReader(raw_store)

        addr = _make_symbol_address(source_id="missing")
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_file(self):
        """Reads full file content for a FILE address."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addr = _make_file_address(source_id="file1", location="module.py")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        assert results[0].content == RAW_FILE_CONTENT
        assert results[0].file_path == "module.py"
        assert results[0].line_range is None

    def test_read_chunk(self):
        """Reads chunk content from address metadata text field."""
        reader = ContentReader(_make_raw_store())

        addr = _make_chunk_address(text="Hello from chunk")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        assert results[0].content == "Hello from chunk"
        assert results[0].file_path == "doc.md"

    def test_read_chunk_empty_text(self):
        """Empty text in chunk metadata results in None (skipped)."""
        reader = ContentReader(_make_raw_store())

        addr = _make_chunk_address(text="")
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_section(self):
        """Reads section content from the section store with metadata."""
        section_data = {
            "sec-001": {
                "content": "Section body text.",
                "title": "Introduction",
                "level": 2,
                "page_start": 1,
                "page_end": 3,
            }
        }
        raw_store = _make_raw_store()
        section_store = _make_section_store(section_data)
        reader = ContentReader(raw_store, section_store=section_store)

        addr = _make_section_address(section_id="sec-001", source_id="file1")
        results = reader.read([addr], limit=10)

        assert len(results) == 1
        r = results[0]
        assert r.content == "Section body text."
        assert r.file_path == "module.py"
        assert r.metadata["section_title"] == "Introduction"
        assert r.metadata["section_level"] == 2
        assert r.metadata["page_start"] == 1
        assert r.metadata["page_end"] == 3

    def test_read_section_no_store(self):
        """When section_store is None, section addresses return None."""
        reader = ContentReader(_make_raw_store(), section_store=None)

        addr = _make_section_address()
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_section_missing_id(self):
        """Missing section_id in metadata results in None."""
        section_store = _make_section_store({"sec-001": {"content": "x"}})
        reader = ContentReader(_make_raw_store(), section_store=section_store)

        addr = Address(
            kind=AddressKind.SECTION,
            source_id="file1",
            location="Section B",
            summary="no id",
            metadata={},  # no section_id key
        )
        results = reader.read([addr], limit=10)

        assert results == []

    def test_read_limits_results(self):
        """Only reads up to `limit` addresses."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addrs = [
            _make_symbol_address(start_line=4, end_line=6),
            _make_file_address(),
        ]
        results = reader.read(addrs, limit=1)

        assert len(results) == 1
        # Only the first address should have been processed
        assert results[0].address is addrs[0]

    def test_read_skips_failed(self):
        """Failed reads are skipped; successful ones are still returned."""
        raw_store = _make_raw_store()
        reader = ContentReader(raw_store)

        addrs = [
            _make_symbol_address(source_id="missing"),  # will fail
            _make_symbol_address(source_id="file1", start_line=4, end_line=5),  # will succeed
        ]
        results = reader.read(addrs, limit=10)

        assert len(results) == 1
        assert results[0].content == "def func():\n    return 42"


# ===========================================================================
# TestCodeExpander
# ===========================================================================


CLASS_FILE_CONTENT = (
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "class MyClass:\n"
    '    """A sample class."""\n'
    "\n"
    "    def __init__(self):\n"
    "        self.x = 1\n"
    "\n"
    "    def do_thing(self):\n"
    "        return self.x\n"
)


NO_IMPORT_CONTENT = "# Just a comment\n" "\n" "def standalone():\n" "    return True\n"


class TestCodeExpander:
    """Tests for CodeExpander."""

    def _make_expander(
        self,
        raw_files: dict[str, dict] | None = None,
        symbol_results: list[dict] | None = None,
        max_expansion_depth: int = 1,
        include_class_context: bool = True,
    ) -> CodeExpander:
        if raw_files is None:
            raw_files = {
                "file1": {
                    "path": "module.py",
                    "content": RAW_FILE_CONTENT,
                }
            }
        raw_store = _make_raw_store(raw_files)
        symbol_store = MagicMock()
        symbol_store.search_by_name.return_value = symbol_results or []
        import_store = MagicMock()
        config = _make_config(
            max_expansion_depth=max_expansion_depth,
            include_class_context=include_class_context,
        )
        return CodeExpander(raw_store, symbol_store, import_store, config)

    def _symbol_read_result(
        self,
        file_path: str = "module.py",
        content: str = "def func():\n    return 42",
        source_id: str = "file1",
        start_line: int = 4,
        end_line: int = 6,
        kind: str = "function",
        qualified_name: str = "mod.func",
    ) -> ReadResult:
        addr = _make_symbol_address(
            source_id=source_id,
            start_line=start_line,
            end_line=end_line,
            kind=kind,
            qualified_name=qualified_name,
        )
        return ReadResult(
            address=addr,
            content=content,
            file_path=file_path,
            line_range=(start_line, end_line),
        )

    def test_expand_adds_imports(self):
        """Symbol result triggers addition of file-level imports."""
        expander = self._make_expander()
        result = self._symbol_read_result()

        expanded = expander.expand([result])

        # Original + imports
        assert len(expanded) == 2
        import_result = expanded[1]
        assert import_result.metadata.get("context_type") == "imports"
        assert "import os" in import_result.content
        assert "import sys" in import_result.content
        assert import_result.file_path == "module.py"

    def test_expand_no_imports(self):
        """File with no imports produces no additional results."""
        raw_files = {
            "file1": {
                "path": "no_imports.py",
                "content": NO_IMPORT_CONTENT,
            }
        }
        expander = self._make_expander(raw_files=raw_files)
        result = self._symbol_read_result(file_path="no_imports.py")

        expanded = expander.expand([result])

        assert len(expanded) == 1
        assert expanded[0] is result

    def test_expand_skips_non_symbol(self):
        """Non-SYMBOL addresses (e.g. SECTION) are not expanded."""
        expander = self._make_expander()
        addr = _make_section_address()
        section_result = ReadResult(
            address=addr,
            content="Section text",
            file_path="doc.md",
        )

        expanded = expander.expand([section_result])

        assert len(expanded) == 1
        assert expanded[0] is section_result

    def test_expand_class_context_for_method(self):
        """Method symbols get a class header added."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        class_symbol = {
            "qualified_name": "module.MyClass",
            "kind": "class",
            "raw_file_id": "file1",
            "start_line": 4,
            "end_line": 12,
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[class_symbol],
        )
        result = self._symbol_read_result(
            content="    def do_thing(self):\n        return self.x",
            start_line=10,
            end_line=12,
            kind="method",
            qualified_name="module.MyClass.do_thing",
        )

        expanded = expander.expand([result])

        # Original + imports + class header = 3
        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert len(class_headers) == 1
        header = class_headers[0]
        assert "class MyClass:" in header.content
        assert header.file_path == "module.py"
        assert header.line_range == (4, 8)  # start_line to min(start+5, end_line)

    def test_expand_no_class_context_for_function(self):
        """Function (not method) symbols do not get a class header."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[],
        )
        result = self._symbol_read_result(
            kind="function",
            qualified_name="mod.func",
        )

        expanded = expander.expand([result])

        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert class_headers == []

    def test_expand_deduplicates(self):
        """Same imports block is not added twice for two symbols in same file."""
        expander = self._make_expander()
        result1 = self._symbol_read_result(start_line=4, end_line=6, qualified_name="mod.func")
        result2 = self._symbol_read_result(
            start_line=7,
            end_line=9,
            qualified_name="mod.other",
            content="def other():\n    pass",
            source_id="file1",
        )

        expanded = expander.expand([result1, result2])

        import_results = [r for r in expanded if r.metadata.get("context_type") == "imports"]
        # The import block duplication is caught either by the inline
        # check in _add_file_imports or by _deduplicate at the end.
        assert len(import_results) == 1

    def test_expand_disabled_when_depth_zero(self):
        """max_expansion_depth=0 returns read results as-is."""
        expander = self._make_expander(max_expansion_depth=0)
        result = self._symbol_read_result()

        expanded = expander.expand([result])

        assert expanded == [result]
        assert len(expanded) == 1

    def test_expand_class_context_disabled(self):
        """include_class_context=False prevents class header expansion."""
        raw_files = {
            "file1": {
                "path": "module.py",
                "content": CLASS_FILE_CONTENT,
            }
        }
        class_symbol = {
            "qualified_name": "module.MyClass",
            "kind": "class",
            "raw_file_id": "file1",
            "start_line": 4,
            "end_line": 12,
        }
        expander = self._make_expander(
            raw_files=raw_files,
            symbol_results=[class_symbol],
            include_class_context=False,
        )
        result = self._symbol_read_result(
            content="    def do_thing(self):\n        return self.x",
            start_line=10,
            end_line=12,
            kind="method",
            qualified_name="module.MyClass.do_thing",
        )

        expanded = expander.expand([result])

        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert class_headers == []
