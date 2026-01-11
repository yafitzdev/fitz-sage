# tests/tabular/test_parser.py
"""Tests for table file parsers."""

import tempfile
from pathlib import Path

import pytest

from fitz_ai.tabular.parser.csv_parser import (
    SUPPORTED_EXTENSIONS,
    can_parse,
    get_sample_rows,
    parse_csv,
)


class TestCanParse:
    """Tests for can_parse function."""

    def test_csv_supported(self):
        """Test that CSV is supported."""
        assert can_parse(Path("data.csv"))
        assert can_parse(Path("DATA.CSV"))

    def test_tsv_supported(self):
        """Test that TSV is supported."""
        assert can_parse(Path("data.tsv"))
        assert can_parse(Path("DATA.TSV"))

    def test_unsupported_extensions(self):
        """Test unsupported extensions."""
        assert not can_parse(Path("data.txt"))
        assert not can_parse(Path("data.json"))
        assert not can_parse(Path("data.xlsx"))
        assert not can_parse(Path("data.pdf"))


class TestParseCSV:
    """Tests for parse_csv function."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory."""
        return tmp_path

    def test_parse_simple_csv(self, temp_dir):
        """Test parsing simple CSV file."""
        csv_content = """Name,Age,City
Alice,30,New York
Bob,25,Los Angeles
Carol,35,Chicago"""

        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = parse_csv(csv_file)

        assert result.columns == ["Name", "Age", "City"]
        assert len(result.rows) == 3
        assert result.rows[0] == ["Alice", "30", "New York"]
        assert result.row_count == 3
        assert result.source_file == str(csv_file)

    def test_parse_tsv(self, temp_dir):
        """Test parsing TSV file."""
        tsv_content = "Name\tAge\tCity\nAlice\t30\tNew York\nBob\t25\tLA"

        tsv_file = temp_dir / "test.tsv"
        tsv_file.write_text(tsv_content)

        result = parse_csv(tsv_file)

        assert result.columns == ["Name", "Age", "City"]
        assert len(result.rows) == 2

    def test_parse_with_quoted_values(self, temp_dir):
        """Test parsing CSV with quoted values."""
        csv_content = '''Name,Description,Value
"John Doe","A very, long description",100
"Jane, Smith","Another ""quoted"" value",200'''

        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        result = parse_csv(csv_file)

        assert result.columns == ["Name", "Description", "Value"]
        assert result.rows[0][0] == "John Doe"
        assert "long description" in result.rows[0][1]

    def test_parse_empty_file_raises(self, temp_dir):
        """Test that empty file raises ValueError."""
        csv_file = temp_dir / "empty.csv"
        csv_file.write_text("")

        with pytest.raises(ValueError, match="Empty table file"):
            parse_csv(csv_file)

    def test_parse_headers_only(self, temp_dir):
        """Test parsing file with headers only."""
        csv_content = "A,B,C"

        csv_file = temp_dir / "headers_only.csv"
        csv_file.write_text(csv_content)

        result = parse_csv(csv_file)

        assert result.columns == ["A", "B", "C"]
        assert result.rows == []
        assert result.row_count == 0

    def test_normalizes_short_rows(self, temp_dir):
        """Test that short rows are padded."""
        csv_content = """A,B,C
1,2,3
4,5
6"""

        csv_file = temp_dir / "short_rows.csv"
        csv_file.write_text(csv_content)

        result = parse_csv(csv_file)

        assert result.rows[0] == ["1", "2", "3"]
        assert result.rows[1] == ["4", "5", ""]  # Padded
        assert result.rows[2] == ["6", "", ""]  # Padded

    def test_truncates_long_rows(self, temp_dir):
        """Test that long rows are truncated."""
        csv_content = """A,B
1,2,3,4
5,6,7"""

        csv_file = temp_dir / "long_rows.csv"
        csv_file.write_text(csv_content)

        result = parse_csv(csv_file)

        assert result.rows[0] == ["1", "2"]  # Truncated
        assert result.rows[1] == ["5", "6"]  # Truncated

    def test_table_id_is_stable(self, temp_dir):
        """Test that table_id is deterministic for same path."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("A,B\n1,2")

        result1 = parse_csv(csv_file)
        result2 = parse_csv(csv_file)

        assert result1.table_id == result2.table_id


class TestGetSampleRows:
    """Tests for get_sample_rows function."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        return tmp_path

    def test_returns_first_n_rows(self, temp_dir):
        """Test getting sample rows."""
        csv_content = "A\n1\n2\n3\n4\n5"
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        parsed = parse_csv(csv_file)
        samples = get_sample_rows(parsed, n=3)

        assert len(samples) == 3
        assert samples[0] == ["1"]
        assert samples[2] == ["3"]

    def test_returns_all_if_less_than_n(self, temp_dir):
        """Test when there are fewer rows than requested."""
        csv_content = "A\n1\n2"
        csv_file = temp_dir / "test.csv"
        csv_file.write_text(csv_content)

        parsed = parse_csv(csv_file)
        samples = get_sample_rows(parsed, n=10)

        assert len(samples) == 2
