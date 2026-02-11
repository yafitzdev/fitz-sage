# tests/unit/test_context_assembler.py
"""Tests for ContextAssembler: formatting, grouping, token budget."""

import pytest

from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_ai.engines.fitz_krag.context.assembler import ContextAssembler
from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult


@pytest.fixture
def config():
    return FitzKragConfig(
        collection="test",
        max_context_tokens=8000,
        include_file_header=True,
        enable_citations=True,
    )


def _make_result(index=1, content="def func(): pass", path="src/mod.py", line_range=(5, 10)):
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id=f"f{index}",
        location=f"mod.func{index}",
        summary=f"Function {index}",
        metadata={"kind": "function"},
    )
    return ReadResult(address=addr, content=content, file_path=path, line_range=line_range)


class TestContextAssembler:
    def test_single_result(self, config):
        assembler = ContextAssembler(config)
        results = [_make_result(1)]
        context = assembler.assemble("test query", results)

        assert "[S1]" in context
        assert "src/mod.py" in context
        assert "lines 5-10" in context
        assert "def func(): pass" in context
        assert "```python" in context

    def test_multiple_results(self, config):
        assembler = ContextAssembler(config)
        results = [_make_result(1), _make_result(2, "class Foo: pass", "src/foo.py", (1, 5))]
        context = assembler.assemble("test", results)

        assert "[S1]" in context
        assert "[S2]" in context
        assert "src/mod.py" in context
        assert "src/foo.py" in context

    def test_empty_results(self, config):
        assembler = ContextAssembler(config)
        context = assembler.assemble("test", [])
        assert context == ""

    def test_respects_token_budget(self):
        config = FitzKragConfig(collection="test", max_context_tokens=100)
        assembler = ContextAssembler(config)
        results = [_make_result(i, "x" * 500, f"file{i}.py", (1, 5)) for i in range(10)]
        context = assembler.assemble("test", results)
        # Should truncate before including all 10 results (100 tokens * 4 chars = 400 char budget)
        assert len(context) < 100 * 4 + 200

    def test_no_file_header(self):
        config = FitzKragConfig(collection="test", include_file_header=False)
        assembler = ContextAssembler(config)
        results = [_make_result(1)]
        context = assembler.assemble("test", results)

        assert "[S1]" in context
        # Should not include file path as header
        assert "# src/mod.py" not in context

    def test_chunk_address_no_language(self, config):
        assembler = ContextAssembler(config)
        addr = Address(kind=AddressKind.CHUNK, source_id="c1", location="doc", summary="")
        result = ReadResult(address=addr, content="some text", file_path="doc.pdf")
        context = assembler.assemble("test", [result])

        # CHUNK should not have language hint
        assert "```\n" in context  # bare code fence without language

    def test_kind_label_in_header(self, config):
        assembler = ContextAssembler(config)
        results = [_make_result(1)]
        context = assembler.assemble("test", results)
        assert "[function]" in context

    def test_section_header_with_pages(self, config):
        assembler = ContextAssembler(config)
        addr = Address(
            kind=AddressKind.SECTION,
            source_id="f1",
            location="Results",
            summary="Results section",
            metadata={"section_id": "sec1", "level": 2},
        )
        result = ReadResult(
            address=addr,
            content="The results show that...",
            file_path="report.pdf",
            metadata={
                "page_start": 12,
                "page_end": 15,
                "section_title": "Results",
                "section_level": 2,
            },
        )
        context = assembler.assemble("test", [result])
        assert "[S1]" in context
        assert "report.pdf" in context
        assert "Results" in context
        assert "pages 12-15" in context
        # SECTION should not have language hint
        assert "```\n" in context

    def test_section_header_single_page(self, config):
        assembler = ContextAssembler(config)
        addr = Address(
            kind=AddressKind.SECTION,
            source_id="f1",
            location="Intro",
            summary="Intro",
            metadata={"section_id": "sec1", "level": 1},
        )
        result = ReadResult(
            address=addr,
            content="Introduction text.",
            file_path="doc.pdf",
            metadata={
                "page_start": 1,
                "page_end": 1,
                "section_title": "Introduction",
                "section_level": 1,
            },
        )
        context = assembler.assemble("test", [result])
        assert "page 1" in context
        assert "pages" not in context
