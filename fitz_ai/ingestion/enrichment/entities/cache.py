# fitz_ai/ingestion/enrichment/entities/cache.py
"""
Caching for extracted entities.

The cache stores LLM-extracted entities keyed by:
- content_hash: Hash of the chunk content
- extractor_id: Identifier of the extractor configuration

This ensures entities are re-extracted when:
1. The content changes (different hash)
2. The extractor changes (different model, prompt version, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .models import Entity

logger = logging.getLogger(__name__)


class EntityCache:
    """
    Persistent cache for extracted entities.

    Caches LLM-extracted entities to avoid redundant API calls.
    Cache keys combine content hash and extractor ID to ensure
    entities are re-extracted when either changes.

    Usage:
        cache = EntityCache(Path(".fitz/entity_cache.json"))

        entities = cache.get(content_hash="abc123", extractor_id="llm:gpt-4o:v1")
        if entities is None:
            entities = extractor.extract(content)
            cache.set(content_hash="abc123", extractor_id="llm:gpt-4o:v1", entities=entities)
    """

    def __init__(self, cache_path: Path | str):
        self._path = Path(cache_path)
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.debug(f"Loaded {len(self._cache)} cached entity extractions")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load entity cache: {e}")
                self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        if not self._dirty:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            self._dirty = False
            logger.debug(f"Saved {len(self._cache)} cached entity extractions")
        except IOError as e:
            logger.warning(f"Failed to save entity cache: {e}")

    @staticmethod
    def _make_key(content_hash: str, extractor_id: str) -> str:
        """Create cache key from content hash and extractor ID."""
        return f"{content_hash}:{extractor_id}"

    def get(self, content_hash: str, extractor_id: str) -> list[Entity] | None:
        """Get cached entities if available."""
        key = self._make_key(content_hash, extractor_id)
        cached = self._cache.get(key)
        if cached is None:
            return None
        return [Entity(**e) for e in cached]

    def set(self, content_hash: str, extractor_id: str, entities: list[Entity]) -> None:
        """Store entities in the cache."""
        key = self._make_key(content_hash, extractor_id)
        self._cache[key] = [e.to_dict() for e in entities]
        self._dirty = True

    def save(self) -> None:
        """Explicitly save cache to disk."""
        self._save()

    def clear(self) -> None:
        """Clear all cached entities."""
        self._cache = {}
        self._dirty = True
        self._save()

    def remove(self, content_hash: str, extractor_id: str) -> bool:
        """Remove a specific entry from the cache."""
        key = self._make_key(content_hash, extractor_id)
        if key in self._cache:
            del self._cache[key]
            self._dirty = True
            return True
        return False

    def __len__(self) -> int:
        """Return number of cached extractions."""
        return len(self._cache)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if a (content_hash, extractor_id) pair is cached."""
        content_hash, extractor_id = key
        return self._make_key(content_hash, extractor_id) in self._cache

    def __enter__(self) -> "EntityCache":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - save on exit."""
        self._save()


__all__ = ["EntityCache"]
