# tests/tabular/test_extractor.py
"""Tests for table extraction during ingestion."""

import pytest

from fitz_ai.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_ai.tabular.extractor import TableExtractor


class TestTableExtractor:
    """Tests for TableExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return TableExtractor()

    def test_extracts_simple_table(self, extractor):
        """Test extracting a simple markdown table."""
        table_content = """| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |"""

        doc = ParsedDocument(
            source="test.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=table_content),
            ],
        )

        modified_doc, table_chunks = extractor.extract(doc)

        # Table should be extracted
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata["is_table_schema"] is True
        assert table_chunks[0].metadata["columns"] == ["Name", "Age"]
        assert table_chunks[0].metadata["row_count"] == 2

        # Modified doc should have no elements (table removed)
        assert len(modified_doc.elements) == 0

    def test_preserves_non_table_elements(self, extractor):
        """Test that non-table elements are preserved."""
        table_content = """| A | B |
|---|---|
| 1 | 2 |"""

        doc = ParsedDocument(
            source="test.pdf",
            elements=[
                DocumentElement(type=ElementType.TEXT, content="Some text"),
                DocumentElement(type=ElementType.TABLE, content=table_content),
                DocumentElement(type=ElementType.HEADING, content="A Heading", level=1),
            ],
        )

        modified_doc, table_chunks = extractor.extract(doc)

        # One table chunk
        assert len(table_chunks) == 1

        # Two non-table elements preserved
        assert len(modified_doc.elements) == 2
        assert modified_doc.elements[0].type == ElementType.TEXT
        assert modified_doc.elements[1].type == ElementType.HEADING

    def test_handles_multiple_tables(self, extractor):
        """Test extracting multiple tables."""
        table1 = """| X | Y |
|---|---|
| a | b |"""

        table2 = """| P | Q | R |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |"""

        doc = ParsedDocument(
            source="multi.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=table1),
                DocumentElement(type=ElementType.TEXT, content="Between tables"),
                DocumentElement(type=ElementType.TABLE, content=table2),
            ],
        )

        modified_doc, table_chunks = extractor.extract(doc)

        # Two table chunks
        assert len(table_chunks) == 2

        # First table: 2 columns, 1 row
        assert table_chunks[0].metadata["columns"] == ["X", "Y"]
        assert table_chunks[0].metadata["row_count"] == 1

        # Second table: 3 columns, 2 rows
        assert table_chunks[1].metadata["columns"] == ["P", "Q", "R"]
        assert table_chunks[1].metadata["row_count"] == 2

        # Text preserved
        assert len(modified_doc.elements) == 1
        assert modified_doc.elements[0].content == "Between tables"

    def test_handles_empty_table(self, extractor):
        """Test handling of empty/malformed tables."""
        # Table with header only, no data rows
        empty_table = """| A | B |
|---|---|"""

        doc = ParsedDocument(
            source="empty.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=empty_table),
            ],
        )

        modified_doc, table_chunks = extractor.extract(doc)

        # Empty table should be kept as regular element (malformed)
        assert len(table_chunks) == 0
        assert len(modified_doc.elements) == 1

    def test_handles_table_without_separator(self, extractor):
        """Test table without separator row (some markdown formats)."""
        table_content = """| Name | Value |
| Item1 | 100 |
| Item2 | 200 |"""

        doc = ParsedDocument(
            source="nosep.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=table_content),
            ],
        )

        modified_doc, table_chunks = extractor.extract(doc)

        # Should still extract the table
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata["columns"] == ["Name", "Value"]
        # Should have 2 data rows (second and third lines)
        assert table_chunks[0].metadata["row_count"] == 2

    def test_generates_stable_table_id(self, extractor):
        """Test that same content generates same table_id."""
        table_content = """| Col | Val |
|-----|-----|
| A | 1 |"""

        doc1 = ParsedDocument(
            source="test1.pdf",
            elements=[DocumentElement(type=ElementType.TABLE, content=table_content)],
        )
        doc2 = ParsedDocument(
            source="test2.pdf",  # Different source
            elements=[DocumentElement(type=ElementType.TABLE, content=table_content)],
        )

        _, chunks1 = extractor.extract(doc1)
        _, chunks2 = extractor.extract(doc2)

        # Same content should produce same table_id (content hash)
        table_id1 = chunks1[0].metadata["table_id"]
        table_id2 = chunks2[0].metadata["table_id"]
        assert table_id1 == table_id2

    def test_preserves_page_metadata(self, extractor):
        """Test that page number is preserved."""
        table_content = """| A | B |
|---|---|
| 1 | 2 |"""

        doc = ParsedDocument(
            source="paged.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=table_content, page=5),
            ],
        )

        _, table_chunks = extractor.extract(doc)

        assert table_chunks[0].metadata["source_page"] == 5

    def test_handles_mismatched_columns(self, extractor):
        """Test handling rows with different column counts."""
        table_content = """| A | B | C |
|---|---|---|
| 1 | 2 |
| 3 | 4 | 5 | 6 |
| 7 | 8 | 9 |"""

        doc = ParsedDocument(
            source="mismatch.pdf",
            elements=[
                DocumentElement(type=ElementType.TABLE, content=table_content),
            ],
        )

        _, table_chunks = extractor.extract(doc)

        # Should handle mismatched rows (pad or truncate)
        assert len(table_chunks) == 1
        # Should have 3 columns (from header)
        assert len(table_chunks[0].metadata["columns"]) == 3


class TestMarkdownParsing:
    """Tests for markdown table parsing edge cases."""

    @pytest.fixture
    def extractor(self):
        return TableExtractor()

    def test_parses_table_with_leading_pipe(self, extractor):
        """Test table with leading pipes."""
        content = """| A | B |
| --- | --- |
| 1 | 2 |"""

        doc = ParsedDocument(
            source="test.pdf",
            elements=[DocumentElement(type=ElementType.TABLE, content=content)],
        )

        _, chunks = extractor.extract(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["columns"] == ["A", "B"]

    def test_parses_table_without_leading_pipe(self, extractor):
        """Test table without leading pipes (valid markdown)."""
        content = """A | B
--- | ---
1 | 2"""

        doc = ParsedDocument(
            source="test.pdf",
            elements=[DocumentElement(type=ElementType.TABLE, content=content)],
        )

        _, chunks = extractor.extract(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["columns"] == ["A", "B"]

    def test_handles_alignment_markers(self, extractor):
        """Test separator with alignment markers (:--, :--:, --:)."""
        content = """| Left | Center | Right |
|:-----|:------:|------:|
| a | b | c |"""

        doc = ParsedDocument(
            source="test.pdf",
            elements=[DocumentElement(type=ElementType.TABLE, content=content)],
        )

        _, chunks = extractor.extract(doc)
        assert len(chunks) == 1
        assert chunks[0].metadata["columns"] == ["Left", "Center", "Right"]
        assert chunks[0].metadata["row_count"] == 1
