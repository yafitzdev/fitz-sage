# tests/unit/test_semantic_math.py
"""
Tests for semantic.py vector math functions.

These tests kill mutation testing survivors by verifying edge cases
that mutations would break.
"""

import pytest

from fitz_sage.core.math import cosine_similarity, mean_vector


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors_returns_one(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_opposite_vectors_returns_negative_one(self):
        """Opposite vectors have similarity -1.0."""
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [-1.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(-1.0)

    def test_orthogonal_vectors_returns_zero(self):
        """Orthogonal vectors have similarity 0.0."""
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)

    def test_zero_vector_a_returns_zero(self):
        """Zero vector A returns 0.0 (not division error).

        KILLS MUTANT: `or` -> `and` on line 129
        """
        vec_a = [0.0, 0.0, 0.0]
        vec_b = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_zero_vector_b_returns_zero(self):
        """Zero vector B returns 0.0 (not division error).

        KILLS MUTANT: `or` -> `and` on line 129
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        """Both zero vectors returns 0.0."""
        vec_a = [0.0, 0.0]
        vec_b = [0.0, 0.0]
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_dimension_mismatch_raises(self):
        """Different dimensions raise ValueError.

        KILLS MUTANT: `!=` -> `==` on line 122
        """
        vec_a = [1.0, 2.0]
        vec_b = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity(vec_a, vec_b)

    def test_same_dimension_no_error(self):
        """Same dimensions don't raise.

        KILLS MUTANT: `!=` -> `==` on line 122
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [4.0, 5.0, 6.0]
        # Should not raise
        result = cosine_similarity(vec_a, vec_b)
        assert isinstance(result, float)

    def test_known_value(self):
        """Test with known computed value.

        KILLS MUTANT: `/` -> `*` on line 132
        """
        # vec_a = [3, 4], vec_b = [4, 3]
        # dot = 12 + 12 = 24
        # norm_a = 5, norm_b = 5
        # similarity = 24 / 25 = 0.96
        vec_a = [3.0, 4.0]
        vec_b = [4.0, 3.0]
        expected = 24.0 / 25.0
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(expected)

    def test_scaling_invariance(self):
        """Cosine similarity is scale-invariant.

        KILLS MUTANT: changes to normalization
        """
        vec_a = [1.0, 2.0, 3.0]
        vec_b = [2.0, 4.0, 6.0]  # vec_a * 2
        # Should be 1.0 (same direction)
        assert cosine_similarity(vec_a, vec_b) == pytest.approx(1.0)


class TestMeanVector:
    """Tests for mean_vector function."""

    def test_single_vector_returns_itself(self):
        """Mean of one vector is itself."""
        vec = [1.0, 2.0, 3.0]
        result = mean_vector([vec])
        assert result == vec

    def test_two_vectors_mean(self):
        """Mean of two vectors.

        KILLS MUTANT: changes to division by n
        """
        vec_a = [0.0, 0.0]
        vec_b = [2.0, 4.0]
        result = mean_vector([vec_a, vec_b])
        assert result == [1.0, 2.0]

    def test_three_vectors_mean(self):
        """Mean of three vectors."""
        vectors = [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
        ]
        result = mean_vector(vectors)
        assert result == [2.0, 4.0]

    def test_empty_list_raises(self):
        """Empty list raises ValueError.

        KILLS MUTANT: removes empty check
        """
        with pytest.raises(ValueError, match="empty"):
            mean_vector([])

    def test_preserves_dimensions(self):
        """Result has same dimensions as input."""
        vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = mean_vector(vectors)
        assert len(result) == 4
