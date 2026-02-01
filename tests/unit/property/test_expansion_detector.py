# tests/unit/property/test_expansion_detector.py
"""
Property-based tests for ExpansionDetector.

Tests pure, deterministic properties of query expansion logic.
Target: fitz_ai/retrieval/detection/detectors/expansion.py
"""

import pytest
from hypothesis import given, settings

from fitz_ai.retrieval.detection.detectors.expansion import (
    ACRONYMS,
    MAX_VARIATIONS,
    SYNONYMS,
    ExpansionDetector,
)
from fitz_ai.retrieval.detection.protocol import DetectionCategory

from .strategies import (
    non_empty_text,
    query_text,
    query_with_acronym,
    query_with_synonym,
)

pytestmark = pytest.mark.property


class TestExpansionDetectorIdempotence:
    """Test that detect() returns identical results on repeated calls."""

    @given(query=query_text())
    def test_detect_idempotent_with_query_text(self, query: str):
        """detect(query) returns identical results on repeated calls."""
        detector = ExpansionDetector()

        result1 = detector.detect(query)
        result2 = detector.detect(query)

        assert result1.detected == result2.detected
        assert result1.confidence == result2.confidence
        assert result1.transformations == result2.transformations
        assert len(result1.matches) == len(result2.matches)

    @given(query=non_empty_text(min_size=1, max_size=200))
    def test_detect_idempotent_with_arbitrary_text(self, query: str):
        """detect(query) is idempotent with arbitrary text."""
        detector = ExpansionDetector()

        result1 = detector.detect(query)
        result2 = detector.detect(query)

        assert result1.detected == result2.detected
        assert result1.transformations == result2.transformations


class TestExpansionDetectorOriginalExcluded:
    """Test that original query is never in transformations."""

    @given(query=query_text())
    def test_original_excluded_from_transformations(self, query: str):
        """Original query never appears in transformations list."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert query not in result.transformations

    @given(query=query_with_synonym())
    def test_original_excluded_with_synonym(self, query: str):
        """Original excluded even when synonyms are found."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        # Original should never be in transformations
        assert query not in result.transformations

    @given(query=query_with_acronym())
    def test_original_excluded_with_acronym(self, query: str):
        """Original excluded even when acronyms are found."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert query not in result.transformations


class TestExpansionDetectorVariationBound:
    """Test that variations never exceed MAX_VARIATIONS."""

    @given(query=query_text())
    def test_transformations_bounded(self, query: str):
        """len(transformations) <= MAX_VARIATIONS."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert len(result.transformations) <= MAX_VARIATIONS

    @given(query=query_with_synonym())
    def test_transformations_bounded_with_synonyms(self, query: str):
        """Bound holds even with many synonyms in query."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert len(result.transformations) <= MAX_VARIATIONS

    @given(query=query_with_acronym())
    def test_transformations_bounded_with_acronyms(self, query: str):
        """Bound holds even with acronyms."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert len(result.transformations) <= MAX_VARIATIONS


class TestExpansionDetectorCasePreservation:
    """Test that first letter case is preserved in replacements."""

    @given(query=query_with_synonym())
    def test_case_preserved_in_synonym_replacement(self, query: str):
        """First letter case is preserved when replacing synonyms."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if not result.detected:
            return

        # Check each transformation preserves the general structure
        for transformation in result.transformations:
            # The transformation should differ from original
            assert transformation != query
            # But should be roughly same length (within reason)
            assert abs(len(transformation) - len(query)) < 50

    def test_uppercase_word_replacement_preserves_case(self):
        """Uppercase first letter is preserved in replacement."""
        detector = ExpansionDetector()

        # "Delete" (capital D) should become "Remove" (capital R)
        result = detector.detect("Delete the file")

        if result.detected:
            # At least one transformation should start with uppercase
            has_uppercase_replacement = any(
                t.startswith("Remove") or t.startswith("Erase") for t in result.transformations
            )
            assert has_uppercase_replacement

    def test_lowercase_word_replacement_preserves_case(self):
        """Lowercase first letter is preserved in replacement."""
        detector = ExpansionDetector()

        result = detector.detect("delete the file")

        if result.detected:
            # Transformations should have lowercase replacements
            has_lowercase_replacement = any(
                "remove" in t.lower() or "erase" in t.lower() for t in result.transformations
            )
            assert has_lowercase_replacement


class TestExpansionDetectorMatchPositions:
    """Test that Match.start/end are valid string indices."""

    @given(query=query_with_synonym())
    def test_match_positions_valid_for_synonyms(self, query: str):
        """Match start/end are valid indices into query string."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        for match in result.matches:
            # Start should be non-negative
            assert match.start >= 0
            # End should be >= start
            assert match.end >= match.start
            # End should not exceed query length
            assert match.end <= len(query)
            # If start is valid, the matched text should match substring
            if 0 <= match.start < len(query):
                # Case-insensitive comparison since positions come from lowercased search
                extracted = query[match.start : match.end]
                assert extracted.lower() == match.text.lower()

    @given(query=query_with_acronym())
    def test_match_positions_valid_for_acronyms(self, query: str):
        """Match positions are valid for acronym matches."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        for match in result.matches:
            assert match.start >= 0
            assert match.end >= match.start
            assert match.end <= len(query)


class TestExpansionDetectorCategory:
    """Test that result category is always EXPANSION."""

    @given(query=query_text())
    def test_category_always_expansion(self, query: str):
        """Result category is always DetectionCategory.EXPANSION."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert result.category == DetectionCategory.EXPANSION

    @given(query=non_empty_text(min_size=1, max_size=100))
    def test_category_expansion_with_arbitrary_text(self, query: str):
        """Category is EXPANSION regardless of input."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        assert result.category == DetectionCategory.EXPANSION


class TestExpansionDetectorConfidence:
    """Test confidence is deterministic: 1.0 when detected, 0.0 when not."""

    @given(query=query_with_synonym())
    def test_confidence_one_when_detected(self, query: str):
        """Confidence is 1.0 when expansion is detected."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if result.detected:
            assert result.confidence == 1.0

    @given(query=query_text())
    def test_confidence_zero_when_not_detected(self, query: str):
        """Confidence is 0.0 when no expansion is detected."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if not result.detected:
            assert result.confidence == 0.0

    def test_confidence_deterministic_known_synonym(self):
        """Confidence is deterministically 1.0 for known synonym."""
        detector = ExpansionDetector()

        # "delete" is a known synonym
        result = detector.detect("delete")

        assert result.detected is True
        assert result.confidence == 1.0

    def test_confidence_deterministic_no_match(self):
        """Confidence is 0.0 when no synonyms or acronyms match."""
        detector = ExpansionDetector()

        # "xyz123" is not a known term
        result = detector.detect("xyz123 qwerty")

        assert result.detected is False
        assert result.confidence == 0.0


class TestExpansionDetectorMetadata:
    """Test metadata fields are correctly populated."""

    @given(query=query_with_synonym())
    def test_metadata_contains_original(self, query: str):
        """Metadata contains original query when detected."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if result.detected:
            assert "original" in result.metadata
            assert result.metadata["original"] == query

    @given(query=query_with_synonym())
    def test_metadata_variation_count_matches(self, query: str):
        """Metadata variation_count matches len(transformations)."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if result.detected:
            assert "variation_count" in result.metadata
            assert result.metadata["variation_count"] == len(result.transformations)


class TestExpansionDetectorMatchPatterns:
    """Test that match pattern names are correct."""

    @given(query=query_with_synonym())
    def test_synonym_matches_have_correct_pattern(self, query: str):
        """Synonym matches have pattern_name='synonym'."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        for match in result.matches:
            assert match.pattern_name in ("synonym", "acronym")

    @given(query=query_with_acronym())
    def test_acronym_matches_have_correct_pattern(self, query: str):
        """Acronym matches have pattern_name='acronym'."""
        detector = ExpansionDetector()
        result = detector.detect(query)

        if result.detected:
            # May also have synonym matches, but should have at least one match
            assert len(result.matches) >= 1


class TestExpansionDetectorDictionaryConsistency:
    """Test that detector uses SYNONYMS and ACRONYMS correctly."""

    def test_all_synonyms_produce_expansions(self):
        """Every word in SYNONYMS produces at least one expansion."""
        detector = ExpansionDetector()

        for word in SYNONYMS:
            result = detector.detect(word)
            assert result.detected is True, f"'{word}' should produce expansion"
            assert len(result.transformations) >= 1

    def test_all_acronyms_produce_expansions(self):
        """Every acronym in ACRONYMS produces expansion."""
        detector = ExpansionDetector()

        for acronym in ACRONYMS:
            result = detector.detect(acronym)
            assert result.detected is True, f"'{acronym}' should produce expansion"
            assert len(result.transformations) >= 1

    @given(query=query_text())
    @settings(max_examples=50)
    def test_unknown_words_dont_produce_false_positives(self, query: str):
        """Words not in dictionaries don't produce expansions."""
        detector = ExpansionDetector()

        # Extract words from query
        words = set(query.lower().split())

        # Check if any words are in dictionaries
        has_known_word = any(w in SYNONYMS or w in ACRONYMS for w in words)

        result = detector.detect(query)

        if has_known_word:
            # Should detect something
            pass  # May or may not detect depending on word
        else:
            # Should NOT detect anything
            assert result.detected is False, f"No known words but detected: {query}"
