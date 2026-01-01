# fitz_ai/ingestion/enrichment/hierarchy/semantic_grouper.py
"""
Semantic chunk grouper using K-means clustering on embeddings.

Groups chunks by semantic similarity rather than metadata keys.
Uses the same interface as ChunkGrouper for seamless integration.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = logging.getLogger(__name__)


def _optimal_k(
    embeddings: np.ndarray,
    max_clusters: int = 10,
    min_clusters: int = 2,
) -> int:
    """
    Find optimal number of clusters using elbow method.

    Args:
        embeddings: (N, D) embedding matrix.
        max_clusters: Maximum k to consider.
        min_clusters: Minimum k to consider.

    Returns:
        Optimal number of clusters.
    """
    from sklearn.cluster import KMeans

    n_samples = len(embeddings)
    if n_samples <= min_clusters:
        return min(min_clusters, n_samples)

    max_k = min(max_clusters, n_samples - 1)
    if max_k < min_clusters:
        return min_clusters

    inertias = []
    k_range = range(min_clusters, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    if len(inertias) < 3:
        return min_clusters

    # Find elbow using second derivative (point of maximum curvature)
    diffs = np.diff(inertias)
    second_diffs = np.diff(diffs)

    if len(second_diffs) > 0:
        elbow_idx = int(np.argmax(second_diffs)) + min_clusters
    else:
        elbow_idx = min_clusters

    optimal = max(min_clusters, min(elbow_idx, max_k))
    logger.debug(f"Optimal k determined: {optimal}")
    return optimal


class SemanticGrouper:
    """
    Groups chunks by embedding similarity using K-means clustering.

    Unlike ChunkGrouper which groups by a metadata key, this groups
    chunks based on the semantic similarity of their embeddings.

    Usage:
        grouper = SemanticGrouper(n_clusters=5)
        groups = grouper.group(chunks, embeddings)
        # groups = {"cluster_0": [...], "cluster_1": [...], ...}
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        max_clusters: int = 10,
        min_cluster_size: int = 2,
    ):
        """
        Initialize semantic grouper.

        Args:
            n_clusters: Fixed number of clusters. If None, auto-detect using elbow method.
            max_clusters: Maximum clusters for auto-detection.
            min_cluster_size: Minimum chunks to form a valid cluster.
        """
        self._n_clusters = n_clusters
        self._max_clusters = max_clusters
        self._min_cluster_size = min_cluster_size

    def group(
        self,
        chunks: List["Chunk"],
        embeddings: np.ndarray,
    ) -> Dict[str, List["Chunk"]]:
        """
        Group chunks by embedding similarity.

        Args:
            chunks: List of chunks to group.
            embeddings: (N, D) matrix of embeddings aligned with chunks.

        Returns:
            Dict mapping cluster_id ("cluster_0", ...) to list of chunks.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for semantic grouping. "
                "Install with: pip install scikit-learn"
            )

        n_samples = len(chunks)

        if n_samples == 0:
            return {}

        if embeddings.shape[0] != n_samples:
            raise ValueError(
                f"Mismatch: {n_samples} chunks but {embeddings.shape[0]} embeddings"
            )

        # Handle small datasets
        if n_samples < self._min_cluster_size:
            logger.info(f"Too few chunks ({n_samples}) for clustering, using single group")
            return {"cluster_0": list(chunks)}

        # Determine number of clusters
        if self._n_clusters is not None:
            n_clusters = min(self._n_clusters, n_samples)
        else:
            n_clusters = _optimal_k(embeddings, max_clusters=self._max_clusters)

        n_clusters = min(n_clusters, n_samples)

        logger.info(f"[SEMANTIC] Clustering {n_samples} chunks into {n_clusters} groups")

        # Run K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(embeddings)

        # Build groups dict
        groups: Dict[str, List["Chunk"]] = defaultdict(list)
        for chunk, label in zip(chunks, labels):
            group_key = f"cluster_{label}"
            groups[group_key].append(chunk)

        # Log cluster sizes
        for key, group_chunks in groups.items():
            logger.debug(f"  {key}: {len(group_chunks)} chunks")

        return dict(groups)


__all__ = ["SemanticGrouper"]
