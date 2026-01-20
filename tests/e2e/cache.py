# tests/e2e/cache.py
"""
File-based response cache for E2E tests.

Each scenario gets its own human-readable .txt file in .e2e_cache/ directory.
To invalidate a specific test's cache, simply delete its file (e.g., E128.txt).

Cache files are invalidated automatically when retrieved chunks change
(tracked via chunk_hash in the file header).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent / ".e2e_cache"


class ResponseCache:
    """
    File-based cache with one human-readable .txt file per scenario.

    Features:
    - Delete individual test cache by removing its .txt file
    - Human-readable format shows query, answer, tier, and chunk IDs
    - Automatic invalidation when chunks change (via hash comparison)
    """

    def __init__(self, max_entries: int = 1000, ttl_days: int = 30, enabled: bool = True):
        """
        Initialize the response cache.

        Args:
            max_entries: Maximum number of cache entries (unused, kept for API compat)
            ttl_days: Days before entries expire (unused, kept for API compat)
            enabled: Whether caching is enabled
        """
        self.max_entries = max_entries
        self.ttl_days = ttl_days
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

        if self.enabled:
            CACHE_DIR.mkdir(exist_ok=True)

    def _make_chunk_hash(self, chunk_ids: list[str]) -> str:
        """Create hash from chunk IDs to detect when chunks change."""
        content = "|".join(sorted(chunk_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cache_path(self, scenario_id: str) -> Path:
        """Get the cache file path for a scenario."""
        return CACHE_DIR / f"{scenario_id}.txt"

    def _parse_cache_file(self, path: Path) -> Optional[dict]:
        """Parse a cache file and return its contents."""
        if not path.exists():
            return None

        try:
            content = path.read_text(encoding="utf-8")

            # Parse header
            chunk_hash_match = re.search(r"^# Chunk Hash: (\w+)", content, re.MULTILINE)
            tier_match = re.search(r"^# Tier: (\w+)", content, re.MULTILINE)
            passed_match = re.search(r"^# Passed: (True|False)", content, re.MULTILINE)

            # Parse sections
            query_match = re.search(r"## Query\n(.+?)(?=\n## )", content, re.DOTALL)
            answer_match = re.search(r"## Answer\n(.+?)(?=\n## |$)", content, re.DOTALL)

            if not all([chunk_hash_match, tier_match, passed_match, query_match, answer_match]):
                logger.debug(f"Cache file {path.name} has invalid format, ignoring")
                return None

            return {
                "chunk_hash": chunk_hash_match.group(1),
                "tier": tier_match.group(1),
                "passed": passed_match.group(1) == "True",
                "query": query_match.group(1).strip(),
                "answer_text": answer_match.group(1).strip(),
            }

        except Exception as e:
            logger.debug(f"Failed to parse cache file {path.name}: {e}")
            return None

    def get(self, query: str, chunk_ids: list[str], scenario_id: str = None) -> Optional[dict]:
        """
        Check cache for existing validated response.

        Args:
            query: The query string
            chunk_ids: List of retrieved chunk IDs (if empty, skips hash validation)
            scenario_id: Test scenario ID (required for file-based cache)

        Returns:
            Dict with answer_text, passed, tier if found and valid, None otherwise
        """
        if not self.enabled or not scenario_id:
            self._misses += 1
            return None

        path = self._get_cache_path(scenario_id)
        cached = self._parse_cache_file(path)

        if not cached:
            self._misses += 1
            return None

        # Verify chunk hash matches (only if chunk_ids provided for comparison)
        # When checking cache before pipeline run, chunk_ids is empty - skip validation
        if chunk_ids:
            current_hash = self._make_chunk_hash(chunk_ids)
            if cached["chunk_hash"] != current_hash:
                logger.debug(
                    f"Cache miss for {scenario_id}: chunks changed "
                    f"(was {cached['chunk_hash']}, now {current_hash})"
                )
                self._misses += 1
                return None

        self._hits += 1
        logger.debug(f"Cache hit for {scenario_id}")

        return {
            "answer_text": cached["answer_text"],
            "passed": cached["passed"],
            "tier": cached["tier"],
            "scenario_id": scenario_id,
        }

    def set(
        self,
        query: str,
        chunk_ids: list[str],
        scenario_id: str,
        answer_text: str,
        passed: bool,
        tier: str,
    ) -> None:
        """
        Store validated response in cache as a human-readable .txt file.

        Args:
            query: The query string
            chunk_ids: List of retrieved chunk IDs
            scenario_id: Test scenario ID
            answer_text: The LLM response
            passed: Whether validation passed
            tier: Which tier produced this result
        """
        if not self.enabled:
            return

        CACHE_DIR.mkdir(exist_ok=True)
        path = self._get_cache_path(scenario_id)
        chunk_hash = self._make_chunk_hash(chunk_ids)
        now = datetime.now(timezone.utc).isoformat()

        # Build human-readable content
        content = f"""# E2E Cache Entry: {scenario_id}
# Created: {now}
# Tier: {tier}
# Passed: {passed}
# Chunk Hash: {chunk_hash}

## Query
{query}

## Answer
{answer_text}

## Chunk IDs
{chr(10).join(sorted(chunk_ids))}
"""

        path.write_text(content, encoding="utf-8")
        logger.debug(f"Cached result for {scenario_id} -> {path.name}")

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        if not self.enabled or not CACHE_DIR.exists():
            return 0

        count = 0
        for path in CACHE_DIR.glob("*.txt"):
            try:
                path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {path.name}: {e}")

        logger.info(f"Cleared {count} cache entries")
        return count

    def clear_scenario(self, scenario_id: str) -> bool:
        """
        Clear cache for a specific scenario.

        Args:
            scenario_id: The scenario ID (e.g., "E128")

        Returns:
            True if file was deleted, False otherwise
        """
        path = self._get_cache_path(scenario_id)
        if path.exists():
            try:
                path.unlink()
                logger.info(f"Cleared cache for {scenario_id}")
                return True
            except Exception as e:
                logger.warning(f"Failed to delete {path.name}: {e}")
        return False

    def stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled or not CACHE_DIR.exists():
            return {"enabled": False, "entries": 0, "hits": 0, "misses": 0}

        entries = list(CACHE_DIR.glob("*.txt"))

        hit_rate = (
            (self._hits / (self._hits + self._misses) * 100)
            if (self._hits + self._misses) > 0
            else 0
        )

        return {
            "enabled": True,
            "entries": len(entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cache_dir": str(CACHE_DIR),
        }

    def list_cached(self) -> list[str]:
        """List all cached scenario IDs."""
        if not CACHE_DIR.exists():
            return []
        return sorted(p.stem for p in CACHE_DIR.glob("*.txt"))
