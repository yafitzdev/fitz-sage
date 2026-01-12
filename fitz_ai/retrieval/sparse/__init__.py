# fitz_ai/retrieval/sparse/__init__.py
"""
Sparse retrieval using TF-IDF for hybrid search.

This module provides sparse (keyword-based) retrieval that complements
dense (semantic) retrieval for hybrid search.
"""

from .index import SparseIndex

__all__ = ["SparseIndex"]
