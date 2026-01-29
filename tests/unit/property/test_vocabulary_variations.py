# tests/unit/property/test_vocabulary_variations.py
"""
Property-based tests for vocabulary variation generation.

Tests pure, deterministic properties of variation generation and normalization.
Target: fitz_ai/retrieval/vocabulary/variations.py
"""

import pytest
from hypothesis import given, assume

from fitz_ai.retrieval.vocabulary.variations import (
    generate_variations,
    normalize_for_matching,
)

from .strategies import (
    testcase_id,
    ticket_id,
    version_string,
    keyword_with_category,
    non_empty_text,
)

pytestmark = pytest.mark.property


class TestGenerateVariationsOriginalIncluded:
    """Test that original keyword is always in result."""

    @given(keyword_category=keyword_with_category())
    def test_original_always_included(self, keyword_category: tuple[str, str]):
        """Original keyword is always in the result list."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        assert keyword in result

    @given(testcase=testcase_id())
    def test_original_included_for_testcase(self, testcase: str):
        """Original testcase ID is in variations."""
        result = generate_variations(testcase, "testcase")
        assert testcase in result

    @given(ticket=ticket_id())
    def test_original_included_for_ticket(self, ticket: str):
        """Original ticket ID is in variations."""
        result = generate_variations(ticket, "ticket")
        assert ticket in result

    @given(version=version_string())
    def test_original_included_for_version(self, version: str):
        """Original version string is in variations."""
        result = generate_variations(version, "version")
        assert version in result


class TestGenerateVariationsNoEmptyStrings:
    """Test that all variations are non-empty."""

    @given(keyword_category=keyword_with_category())
    def test_no_empty_variations(self, keyword_category: tuple[str, str]):
        """All variations are non-empty strings."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        for variation in result:
            assert variation, f"Empty variation found for '{keyword}'"
            assert len(variation) > 0

    @given(testcase=testcase_id())
    def test_no_empty_for_testcase(self, testcase: str):
        """No empty variations for testcase IDs."""
        result = generate_variations(testcase, "testcase")
        assert all(v for v in result)

    @given(ticket=ticket_id())
    def test_no_empty_for_ticket(self, ticket: str):
        """No empty variations for ticket IDs."""
        result = generate_variations(ticket, "ticket")
        assert all(v for v in result)


class TestGenerateVariationsNoDuplicates:
    """Test that all variations are unique."""

    @given(keyword_category=keyword_with_category())
    def test_no_duplicate_variations(self, keyword_category: tuple[str, str]):
        """All variations are unique (no duplicates)."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        assert len(result) == len(set(result)), f"Duplicates found for '{keyword}'"

    @given(testcase=testcase_id())
    def test_no_duplicates_for_testcase(self, testcase: str):
        """No duplicate variations for testcase IDs."""
        result = generate_variations(testcase, "testcase")
        assert len(result) == len(set(result))

    @given(ticket=ticket_id())
    def test_no_duplicates_for_ticket(self, ticket: str):
        """No duplicate variations for ticket IDs."""
        result = generate_variations(ticket, "ticket")
        assert len(result) == len(set(result))


class TestGenerateVariationsSorted:
    """Test that result is sorted case-insensitively."""

    @given(keyword_category=keyword_with_category())
    def test_result_is_sorted(self, keyword_category: tuple[str, str]):
        """Result is sorted case-insensitively."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)
        expected = sorted(result, key=str.lower)

        assert result == expected, f"Not sorted for '{keyword}'"

    @given(testcase=testcase_id())
    def test_sorted_for_testcase(self, testcase: str):
        """Result is sorted for testcase IDs."""
        result = generate_variations(testcase, "testcase")
        expected = sorted(result, key=str.lower)
        assert result == expected

    @given(ticket=ticket_id())
    def test_sorted_for_ticket(self, ticket: str):
        """Result is sorted for ticket IDs."""
        result = generate_variations(ticket, "ticket")
        expected = sorted(result, key=str.lower)
        assert result == expected


class TestGenerateVariationsCaseVariations:
    """Test that lowercase and uppercase variations exist."""

    @given(keyword_category=keyword_with_category())
    def test_lowercase_exists(self, keyword_category: tuple[str, str]):
        """keyword.lower() is in variations."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        assert keyword.lower() in result

    @given(keyword_category=keyword_with_category())
    def test_uppercase_exists(self, keyword_category: tuple[str, str]):
        """keyword.upper() is in variations."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        assert keyword.upper() in result

    @given(testcase=testcase_id())
    def test_case_variations_for_testcase(self, testcase: str):
        """Both case variations exist for testcase."""
        result = generate_variations(testcase, "testcase")
        assert testcase.lower() in result
        assert testcase.upper() in result

    @given(ticket=ticket_id())
    def test_case_variations_for_ticket(self, ticket: str):
        """Both case variations exist for ticket."""
        result = generate_variations(ticket, "ticket")
        assert ticket.lower() in result
        assert ticket.upper() in result


class TestGenerateVariationsDeterminism:
    """Test that generate_variations is deterministic."""

    @given(keyword_category=keyword_with_category())
    def test_deterministic_output(self, keyword_category: tuple[str, str]):
        """Same input produces same output."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result1 = generate_variations(keyword, category)
        result2 = generate_variations(keyword, category)

        assert result1 == result2


class TestGenerateVariationsMinimumCount:
    """Test that at least some variations are generated."""

    @given(keyword_category=keyword_with_category())
    def test_at_least_one_variation(self, keyword_category: tuple[str, str]):
        """At least 1 variation is generated (the original)."""
        keyword, category = keyword_category
        assume(len(keyword.strip()) > 0)

        result = generate_variations(keyword, category)

        assert len(result) >= 1

    @given(testcase=testcase_id())
    def test_testcase_generates_many_variations(self, testcase: str):
        """Testcase IDs generate multiple variations."""
        result = generate_variations(testcase, "testcase")
        # Should have original + case variations + category-specific
        assert len(result) >= 3

    @given(ticket=ticket_id())
    def test_ticket_generates_variations(self, ticket: str):
        """Ticket IDs generate multiple variations."""
        result = generate_variations(ticket, "ticket")
        assert len(result) >= 3


class TestNormalizeForMatchingIdempotence:
    """Test that normalize is idempotent."""

    @given(text=non_empty_text(min_size=1, max_size=100))
    def test_normalize_idempotent(self, text: str):
        """normalize(normalize(x)) == normalize(x)."""
        once = normalize_for_matching(text)
        twice = normalize_for_matching(once)

        assert once == twice

    @given(testcase=testcase_id())
    def test_normalize_idempotent_for_testcase(self, testcase: str):
        """Normalize is idempotent for testcase IDs."""
        once = normalize_for_matching(testcase)
        twice = normalize_for_matching(once)
        assert once == twice

    @given(ticket=ticket_id())
    def test_normalize_idempotent_for_ticket(self, ticket: str):
        """Normalize is idempotent for ticket IDs."""
        once = normalize_for_matching(ticket)
        twice = normalize_for_matching(once)
        assert once == twice


class TestNormalizeForMatchingLowercase:
    """Test that normalize always returns lowercase."""

    @given(text=non_empty_text(min_size=1, max_size=100))
    def test_result_is_lowercase(self, text: str):
        """Result is always lowercase."""
        result = normalize_for_matching(text)

        assert result == result.lower()

    @given(testcase=testcase_id())
    def test_lowercase_for_testcase(self, testcase: str):
        """Result is lowercase for testcase."""
        result = normalize_for_matching(testcase)
        assert result == result.lower()

    @given(ticket=ticket_id())
    def test_lowercase_for_ticket(self, ticket: str):
        """Result is lowercase for ticket."""
        result = normalize_for_matching(ticket)
        assert result == result.lower()


class TestNormalizeForMatchingStripped:
    """Test that normalize removes leading/trailing whitespace."""

    @given(text=non_empty_text(min_size=1, max_size=100))
    def test_result_is_stripped(self, text: str):
        """Result has no leading/trailing whitespace."""
        result = normalize_for_matching(text)

        assert result == result.strip()

    def test_strips_leading_whitespace(self):
        """Leading whitespace is removed."""
        result = normalize_for_matching("  hello world")
        assert result == "hello world"

    def test_strips_trailing_whitespace(self):
        """Trailing whitespace is removed."""
        result = normalize_for_matching("hello world  ")
        assert result == "hello world"

    def test_strips_both(self):
        """Both leading and trailing whitespace removed."""
        result = normalize_for_matching("  hello world  ")
        assert result == "hello world"


class TestNormalizeForMatchingSeparators:
    """Test that separators are normalized to single spaces."""

    def test_hyphen_to_space(self):
        """Hyphens become spaces."""
        result = normalize_for_matching("hello-world")
        assert result == "hello world"

    def test_underscore_to_space(self):
        """Underscores become spaces."""
        result = normalize_for_matching("hello_world")
        assert result == "hello world"

    def test_multiple_separators_collapse(self):
        """Multiple separators collapse to single space."""
        result = normalize_for_matching("hello---world")
        assert result == "hello world"

        result = normalize_for_matching("hello___world")
        assert result == "hello world"

        result = normalize_for_matching("hello   world")
        assert result == "hello world"

    def test_mixed_separators(self):
        """Mixed separators all become single space."""
        result = normalize_for_matching("hello-_- world")
        assert result == "hello world"

    @given(testcase=testcase_id())
    def test_separators_normalized_for_testcase(self, testcase: str):
        """Separators normalized in testcase IDs."""
        result = normalize_for_matching(testcase)

        # Should not have consecutive spaces, hyphens, or underscores
        assert "--" not in result
        assert "__" not in result
        assert "  " not in result


class TestNormalizeForMatchingConsistency:
    """Test that equivalent inputs produce same output."""

    def test_separator_equivalence(self):
        """Different separators produce same normalized form."""
        inputs = [
            "hello-world",
            "hello_world",
            "hello world",
            "hello - world",
            "hello _ world",
        ]

        results = [normalize_for_matching(i) for i in inputs]

        # All should produce the same result
        assert len(set(results)) == 1
        assert results[0] == "hello world"

    def test_case_equivalence(self):
        """Different cases produce same normalized form."""
        inputs = [
            "Hello World",
            "hello world",
            "HELLO WORLD",
            "HeLLo WoRLd",
        ]

        results = [normalize_for_matching(i) for i in inputs]

        assert len(set(results)) == 1
        assert results[0] == "hello world"

    @given(text=non_empty_text(min_size=3, max_size=50))
    def test_normalize_deterministic(self, text: str):
        """Same input always produces same output."""
        result1 = normalize_for_matching(text)
        result2 = normalize_for_matching(text)

        assert result1 == result2
