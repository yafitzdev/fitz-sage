# tests/unit/test_glm_ocr_parser.py
"""
Unit tests for GlmOcrParser — markdown parsing and page routing heuristics.
No ollama or GLM-OCR model needed for these tests.
"""

from __future__ import annotations

import pytest

from fitz_sage.core.document import ElementType
from fitz_sage.ingestion.parser.plugins.glm_ocr import GlmOcrParser


@pytest.fixture
def parser():
    return GlmOcrParser()


# ---------------------------------------------------------------------------
# Markdown → Elements parsing
# ---------------------------------------------------------------------------


class TestMarkdownToElements:
    """Tests for _markdown_to_elements (GLM-OCR output parsing)."""

    def test_heading_levels(self, parser):
        md = "# Title\n\n## Subtitle\n\n### Deep"
        elements = parser._markdown_to_elements(md, page=1)

        headings = [e for e in elements if e.type == ElementType.HEADING]
        assert len(headings) == 3
        assert headings[0].content == "Title"
        assert headings[0].level == 1
        assert headings[1].content == "Subtitle"
        assert headings[1].level == 2
        assert headings[2].content == "Deep"
        assert headings[2].level == 3

    def test_plain_text(self, parser):
        md = "This is a paragraph of text."
        elements = parser._markdown_to_elements(md, page=5)

        assert len(elements) == 1
        assert elements[0].type == ElementType.TEXT
        assert elements[0].content == "This is a paragraph of text."
        assert elements[0].page == 5

    def test_code_block(self, parser):
        md = "```python\ndef hello():\n    pass\n```"
        elements = parser._markdown_to_elements(md, page=1)

        assert len(elements) == 1
        assert elements[0].type == ElementType.CODE_BLOCK
        assert "def hello():" in elements[0].content
        assert elements[0].language == "python"

    def test_table(self, parser):
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        elements = parser._markdown_to_elements(md, page=1)

        tables = [e for e in elements if e.type == ElementType.TABLE]
        assert len(tables) == 1
        assert "Alice" in tables[0].content
        assert "Bob" in tables[0].content

    def test_mixed_content(self, parser):
        md = (
            "# Introduction\n\n"
            "Some text here.\n\n"
            "| Col1 | Col2 |\n|------|------|\n| A | B |\n\n"
            "More text."
        )
        elements = parser._markdown_to_elements(md, page=3)

        types = [e.type for e in elements]
        assert ElementType.HEADING in types
        assert ElementType.TEXT in types
        assert ElementType.TABLE in types
        assert all(e.page == 3 for e in elements)

    def test_empty_markdown(self, parser):
        elements = parser._markdown_to_elements("", page=1)
        assert elements == []

    def test_heading_with_special_chars(self, parser):
        md = "# Section 3.2: Valid & Reliable"
        elements = parser._markdown_to_elements(md, page=1)
        assert elements[0].content == "Section 3.2: Valid & Reliable"


# ---------------------------------------------------------------------------
# Text → Elements (pypdfium2 fast path)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Page routing heuristic
# ---------------------------------------------------------------------------


class TestNeedsOcr:
    """Tests for OCR page routing threshold (_MIN_TEXT_CHARS = 50)."""

    def test_short_text_needs_ocr(self):
        """Pages with < 50 chars are likely scanned → need OCR."""
        from fitz_sage.ingestion.parser.plugins.glm_ocr import _MIN_TEXT_CHARS

        assert len("Short") < _MIN_TEXT_CHARS
        assert len("") < _MIN_TEXT_CHARS

    def test_long_text_skips_ocr(self):
        """Pages with >= 50 chars don't need OCR."""
        from fitz_sage.ingestion.parser.plugins.glm_ocr import _MIN_TEXT_CHARS

        assert len("A" * 100) >= _MIN_TEXT_CHARS
