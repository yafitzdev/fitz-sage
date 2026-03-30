# fitz_sage/core/math.py
"""
Vector math utilities - Pure stdlib math, no fitz imports.

Extracted from guardrails/semantic.py for shared use across engines.
"""

from __future__ import annotations

import math


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value in [-1, 1] where 1 means identical direction.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of vectors."""
    if not vectors:
        raise ValueError("Cannot compute mean of empty vector list")

    dim = len(vectors[0])
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    n = len(vectors)
    return [v / n for v in result]


__all__ = ["cosine_similarity", "mean_vector"]
