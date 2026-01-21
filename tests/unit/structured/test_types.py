# tests/unit/structured/test_types.py
"""Tests for structured data type inference and coercion."""

from __future__ import annotations

from datetime import datetime

from fitz_ai.structured.types import (
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_NUMBER,
    TYPE_STRING,
    coerce_value,
    infer_column_type,
    infer_type,
    is_indexable_column,
    select_indexed_columns,
)


class TestInferType:
    """Tests for single value type inference."""

    def test_infer_none_returns_string(self):
        """None defaults to string."""
        assert infer_type(None) == TYPE_STRING

    def test_infer_bool_true(self):
        """Boolean True is detected."""
        assert infer_type(True) == TYPE_BOOLEAN

    def test_infer_bool_false(self):
        """Boolean False is detected."""
        assert infer_type(False) == TYPE_BOOLEAN

    def test_infer_int(self):
        """Integer is detected as number."""
        assert infer_type(42) == TYPE_NUMBER
        assert infer_type(-100) == TYPE_NUMBER
        assert infer_type(0) == TYPE_NUMBER

    def test_infer_float(self):
        """Float is detected as number."""
        assert infer_type(3.14) == TYPE_NUMBER
        assert infer_type(-0.5) == TYPE_NUMBER

    def test_infer_datetime(self):
        """datetime object is detected as date."""
        assert infer_type(datetime.now()) == TYPE_DATE

    def test_infer_string_boolean_values(self):
        """String boolean values are detected."""
        assert infer_type("true") == TYPE_BOOLEAN
        assert infer_type("True") == TYPE_BOOLEAN
        assert infer_type("TRUE") == TYPE_BOOLEAN
        assert infer_type("false") == TYPE_BOOLEAN
        assert infer_type("yes") == TYPE_BOOLEAN
        assert infer_type("no") == TYPE_BOOLEAN
        assert infer_type("1") == TYPE_BOOLEAN  # Also matches number but bool checked first
        assert infer_type("0") == TYPE_BOOLEAN

    def test_infer_string_number_values(self):
        """String number values are detected."""
        assert infer_type("42") == TYPE_NUMBER
        assert infer_type("3.14") == TYPE_NUMBER
        assert infer_type("-100") == TYPE_NUMBER
        assert infer_type("1,000") == TYPE_NUMBER  # Comma separator
        assert infer_type("1,234.56") == TYPE_NUMBER

    def test_infer_string_date_iso(self):
        """ISO date strings are detected."""
        assert infer_type("2024-01-15") == TYPE_DATE
        assert infer_type("2024-01-15T10:30:00") == TYPE_DATE

    def test_infer_string_date_us_format(self):
        """US date format is detected."""
        assert infer_type("1/15/2024") == TYPE_DATE
        assert infer_type("12/31/2024") == TYPE_DATE

    def test_infer_string_date_eu_format(self):
        """EU date format is detected."""
        assert infer_type("15-01-2024") == TYPE_DATE

    def test_infer_plain_string(self):
        """Plain strings are detected."""
        assert infer_type("hello world") == TYPE_STRING
        assert infer_type("John Doe") == TYPE_STRING
        assert infer_type("engineering") == TYPE_STRING

    def test_infer_empty_string(self):
        """Empty string is detected as string."""
        assert infer_type("") == TYPE_STRING
        assert infer_type("  ") == TYPE_STRING


class TestInferColumnType:
    """Tests for column type inference from multiple values."""

    def test_empty_values_returns_string(self):
        """Empty list defaults to string."""
        assert infer_column_type([]) == TYPE_STRING

    def test_all_nulls_returns_string(self):
        """All nulls defaults to string."""
        assert infer_column_type([None, None, None]) == TYPE_STRING
        assert infer_column_type(["", "", ""]) == TYPE_STRING

    def test_majority_numbers(self):
        """Column with mostly numbers is detected."""
        values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        assert infer_column_type(values) == TYPE_NUMBER

    def test_majority_dates(self):
        """Column with mostly dates is detected."""
        values = [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2024-01-07",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        ]
        assert infer_column_type(values) == TYPE_DATE

    def test_majority_booleans(self):
        """Column with mostly booleans is detected."""
        values = ["yes", "no", "yes", "yes", "no", "yes", "no", "yes", "yes", "no"]
        assert infer_column_type(values) == TYPE_BOOLEAN

    def test_mixed_types_fallback_to_string(self):
        """Mixed types fall back to string."""
        values = ["hello", 42, "2024-01-01", True, "world"]
        assert infer_column_type(values) == TYPE_STRING

    def test_nulls_ignored_in_inference(self):
        """Null values are ignored when inferring type."""
        values = [None, 100, None, 200, None, 300, None, 400, 500, 600]
        assert infer_column_type(values) == TYPE_NUMBER


class TestCoerceValue:
    """Tests for value coercion."""

    def test_coerce_none_returns_none(self):
        """None stays None regardless of target type."""
        assert coerce_value(None, TYPE_STRING) is None
        assert coerce_value(None, TYPE_NUMBER) is None
        assert coerce_value(None, TYPE_BOOLEAN) is None
        assert coerce_value(None, TYPE_DATE) is None

    def test_coerce_to_string(self):
        """Values coerce to string."""
        assert coerce_value(42, TYPE_STRING) == "42"
        assert coerce_value(3.14, TYPE_STRING) == "3.14"
        assert coerce_value(True, TYPE_STRING) == "True"

    def test_coerce_to_number_from_int(self):
        """Int stays int when coerced to number."""
        assert coerce_value(42, TYPE_NUMBER) == 42

    def test_coerce_to_number_from_string(self):
        """String number coerces to number."""
        assert coerce_value("42", TYPE_NUMBER) == 42
        assert coerce_value("3.14", TYPE_NUMBER) == 3.14
        assert coerce_value("1,000", TYPE_NUMBER) == 1000

    def test_coerce_to_number_invalid_returns_none(self):
        """Invalid number string returns None."""
        assert coerce_value("hello", TYPE_NUMBER) is None

    def test_coerce_to_boolean_from_bool(self):
        """Bool stays bool."""
        assert coerce_value(True, TYPE_BOOLEAN) is True
        assert coerce_value(False, TYPE_BOOLEAN) is False

    def test_coerce_to_boolean_from_string(self):
        """String booleans coerce."""
        assert coerce_value("true", TYPE_BOOLEAN) is True
        assert coerce_value("yes", TYPE_BOOLEAN) is True
        assert coerce_value("false", TYPE_BOOLEAN) is False
        assert coerce_value("no", TYPE_BOOLEAN) is False

    def test_coerce_to_boolean_invalid_returns_none(self):
        """Invalid boolean string returns None."""
        assert coerce_value("maybe", TYPE_BOOLEAN) is None

    def test_coerce_to_date_from_datetime(self):
        """datetime coerces to ISO string."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = coerce_value(dt, TYPE_DATE)
        assert result == "2024-01-15T10:30:00"

    def test_coerce_to_date_from_string(self):
        """Date string stays as string."""
        assert coerce_value("2024-01-15", TYPE_DATE) == "2024-01-15"


class TestIsIndexableColumn:
    """Tests for column indexability heuristics."""

    def test_primary_key_always_indexed(self):
        """Primary key is always indexed."""
        assert is_indexable_column(TYPE_STRING, 1.0, is_primary_key=True)
        assert is_indexable_column(TYPE_NUMBER, 1.0, is_primary_key=True)

    def test_boolean_always_indexed(self):
        """Boolean columns are always indexed."""
        assert is_indexable_column(TYPE_BOOLEAN, 0.5)
        assert is_indexable_column(TYPE_BOOLEAN, 0.1)

    def test_date_always_indexed(self):
        """Date columns are always indexed."""
        assert is_indexable_column(TYPE_DATE, 0.9)
        assert is_indexable_column(TYPE_DATE, 0.5)

    def test_number_indexed_when_moderate_cardinality(self):
        """Numbers indexed when not too unique or constant."""
        assert is_indexable_column(TYPE_NUMBER, 0.1)  # Low cardinality
        assert is_indexable_column(TYPE_NUMBER, 0.5)  # Moderate
        assert not is_indexable_column(TYPE_NUMBER, 0.95)  # Too unique
        assert not is_indexable_column(TYPE_NUMBER, 0.001)  # Almost constant

    def test_string_indexed_when_low_cardinality(self):
        """Strings indexed when they look like enums."""
        assert is_indexable_column(TYPE_STRING, 0.1)  # Enum-like
        assert is_indexable_column(TYPE_STRING, 0.05)  # Enum-like
        assert not is_indexable_column(TYPE_STRING, 0.9)  # Too unique
        assert not is_indexable_column(TYPE_STRING, 1.0)  # All unique


class TestSelectIndexedColumns:
    """Tests for auto-selecting indexed columns."""

    def test_primary_key_always_included(self):
        """Primary key is always in indexed list."""
        columns = ["id", "name", "email"]
        types = [TYPE_STRING, TYPE_STRING, TYPE_STRING]
        samples = [["1", "2", "3"], ["John", "Jane", "Bob"], ["a@b.com", "c@d.com", "e@f.com"]]

        indexed = select_indexed_columns(columns, types, samples, "id")

        assert "id" in indexed

    def test_boolean_and_date_selected(self):
        """Boolean and date columns are auto-selected."""
        columns = ["id", "is_active", "created_at", "name"]
        types = [TYPE_STRING, TYPE_BOOLEAN, TYPE_DATE, TYPE_STRING]
        samples = [
            ["1", "2", "3"],
            [True, False, True],
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            ["John", "Jane", "Bob"],
        ]

        indexed = select_indexed_columns(columns, types, samples, "id")

        assert "id" in indexed
        assert "is_active" in indexed
        assert "created_at" in indexed

    def test_max_indexed_limit(self):
        """Respects max indexed columns limit."""
        columns = ["id", "a", "b", "c", "d", "e", "f", "g"]
        types = [TYPE_STRING] + [TYPE_BOOLEAN] * 7  # All indexable
        samples = [[str(i) for i in range(10)] for _ in columns]

        indexed = select_indexed_columns(columns, types, samples, "id", max_indexed=3)

        # Should have id + max 3 others
        assert len(indexed) <= 4

    def test_enum_like_strings_selected(self):
        """String columns with low cardinality are selected."""
        columns = ["id", "department", "name"]
        types = [TYPE_STRING, TYPE_STRING, TYPE_STRING]
        samples = [
            ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            # Low cardinality: 2 unique values out of 10 = 0.2 ratio (< 0.3 threshold)
            ["eng", "eng", "sales", "eng", "sales", "eng", "eng", "eng", "sales", "eng"],
            ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],  # High cardinality
        ]

        indexed = select_indexed_columns(columns, types, samples, "id")

        assert "id" in indexed
        assert "department" in indexed
        assert "name" not in indexed
