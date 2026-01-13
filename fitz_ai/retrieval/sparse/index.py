# fitz_ai/retrieval/sparse/index.py
"""
Sparse Index for hybrid retrieval.

Uses TF-IDF vectorization for keyword-based retrieval.
Complements dense (semantic) search with exact keyword matching.

Requires: scikit-learn, scipy (optional - gracefully degrades if not available)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Optional dependencies - sparse search degrades gracefully if not available
try:
    import numpy as np
    from scipy.sparse import csr_matrix, load_npz, save_npz
    from sklearn.feature_extraction.text import TfidfVectorizer

    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False
    np = None
    csr_matrix = None
    save_npz = None
    load_npz = None
    TfidfVectorizer = None


@dataclass
class SparseHit:
    """A single sparse search result."""

    chunk_id: str
    score: float


class SparseIndex:
    """
    TF-IDF based sparse index for hybrid retrieval.

    Stores:
    - TF-IDF vectorizer (vocabulary + IDF weights)
    - Document vectors (sparse matrix)
    - Chunk ID mapping

    Usage (build):
        index = SparseIndex(collection="my_collection")
        index.build(chunk_ids=["c1", "c2"], contents=["hello world", "foo bar"])
        index.save()

    Usage (query):
        index = SparseIndex.load(collection="my_collection")
        hits = index.search("hello", k=10)
    """

    def __init__(self, collection: str):
        """
        Initialize sparse index for a collection.

        Args:
            collection: Collection name (used for storage path)
        """
        self.collection = collection
        self.vectorizer: Any = None  # TfidfVectorizer when available
        self.doc_vectors: Any = None  # csr_matrix when available
        self.chunk_ids: list[str] = []

    @classmethod
    def load(cls, collection: str) -> "SparseIndex":
        """
        Load a sparse index from disk.

        Args:
            collection: Collection name

        Returns:
            Loaded SparseIndex, or empty index if not found
        """
        index = cls(collection=collection)

        if not SPARSE_AVAILABLE:
            logger.debug("Sparse index not available (scipy/sklearn not installed)")
            return index

        base_path = FitzPaths.sparse_index(collection)
        vectorizer_path = base_path.with_suffix(".pkl")
        vectors_path = base_path.with_suffix(".npz")
        ids_path = base_path.with_suffix(".json")

        if not vectorizer_path.exists():
            logger.debug(f"No sparse index found for collection '{collection}'")
            return index

        try:
            # Load vectorizer
            with open(vectorizer_path, "rb") as f:
                index.vectorizer = pickle.load(f)

            # Load sparse matrix
            index.doc_vectors = load_npz(vectors_path)

            # Load chunk IDs
            with open(ids_path, "r", encoding="utf-8") as f:
                index.chunk_ids = json.load(f)

            logger.debug(
                f"Loaded sparse index for '{collection}': " f"{len(index.chunk_ids)} documents"
            )

        except Exception as e:
            logger.warning(f"Failed to load sparse index: {e}")
            # Return empty index
            index.vectorizer = None
            index.doc_vectors = None
            index.chunk_ids = []

        return index

    def build(self, chunk_ids: list[str], contents: list[str]) -> None:
        """
        Build the sparse index from chunks.

        Args:
            chunk_ids: List of chunk IDs
            contents: List of chunk contents (same order as chunk_ids)
        """
        if not SPARSE_AVAILABLE:
            logger.debug("Sparse index not available (scipy/sklearn not installed)")
            return

        if len(chunk_ids) != len(contents):
            raise ValueError("chunk_ids and contents must have same length")

        if not chunk_ids:
            logger.debug("No chunks to index")
            return

        # Configure TF-IDF vectorizer
        # - sublinear_tf: Use log(tf) for better weighting
        # - min_df: Ignore terms appearing in fewer than 2 docs
        # - max_df: Ignore terms appearing in >95% of docs (stopwords)
        # - ngram_range: Include bigrams for phrase matching
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=1,  # Keep all terms for small corpora
            max_df=0.95,
            ngram_range=(1, 2),  # Unigrams and bigrams
            lowercase=True,
            strip_accents="unicode",
        )

        # Fit and transform
        self.doc_vectors = self.vectorizer.fit_transform(contents)
        self.chunk_ids = list(chunk_ids)

        logger.debug(
            f"Built sparse index: {len(self.chunk_ids)} documents, "
            f"{len(self.vectorizer.vocabulary_)} terms"
        )

    def add(self, chunk_ids: list[str], contents: list[str]) -> None:
        """
        Add new chunks to existing index.

        For incremental ingestion. Rebuilds the full index (TF-IDF requires this).

        Args:
            chunk_ids: New chunk IDs
            contents: New chunk contents
        """
        if not self.vectorizer or self.doc_vectors is None:
            # No existing index, just build
            self.build(chunk_ids, contents)
            return

        # For TF-IDF, we need to refit on all documents
        # (IDF weights change when corpus changes)
        # This is a known limitation - for large corpora, consider BM25 with
        # pre-computed IDF or approximate methods

        # Get existing contents by loading from vector DB would be expensive
        # Instead, just rebuild with new chunks only
        # This means IDF weights are only for new chunks, which is suboptimal
        # but acceptable for incremental updates

        # For now, just extend without recomputing IDF
        new_vectors = self.vectorizer.transform(contents)

        # Stack sparse matrices
        from scipy.sparse import vstack

        self.doc_vectors = vstack([self.doc_vectors, new_vectors])
        self.chunk_ids.extend(chunk_ids)

        logger.debug(f"Added {len(chunk_ids)} chunks to sparse index")

    def save(self) -> None:
        """Save the sparse index to disk."""
        if not self.vectorizer or self.doc_vectors is None:
            logger.debug("No index to save")
            return

        base_path = FitzPaths.sparse_index(self.collection)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        vectorizer_path = base_path.with_suffix(".pkl")
        vectors_path = base_path.with_suffix(".npz")
        ids_path = base_path.with_suffix(".json")

        # Save vectorizer
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        # Save sparse matrix
        save_npz(vectors_path, self.doc_vectors)

        # Save chunk IDs
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)

        logger.debug(f"Saved sparse index to {base_path}")

    def search(self, query: str, k: int = 10) -> list[SparseHit]:
        """
        Search the sparse index.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of SparseHit with chunk_id and score
        """
        if not self.vectorizer or self.doc_vectors is None:
            return []

        # Transform query
        query_vector = self.vectorizer.transform([query])

        # Compute cosine similarity (TF-IDF vectors are L2-normalized)
        scores = (self.doc_vectors @ query_vector.T).toarray().flatten()

        # Get top-k indices
        if len(scores) <= k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        # Filter zero scores
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append(
                    SparseHit(
                        chunk_id=self.chunk_ids[idx],
                        score=score,
                    )
                )

        return results

    def is_ready(self) -> bool:
        """Check if index is ready for querying."""
        return (
            self.vectorizer is not None and self.doc_vectors is not None and len(self.chunk_ids) > 0
        )

    def __len__(self) -> int:
        """Number of indexed documents."""
        return len(self.chunk_ids)
