# tests/e2e/cache.py
"""Response cache for e2e tests."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

CACHE_DB = Path(__file__).parent / ".e2e_cache.db"


class ResponseCache:
    """
    Cache validated responses by query + chunk hash.

    This allows skipping LLM calls when the same query produces the same
    retrieval results as a previously validated run.
    """

    def __init__(self, max_entries: int = 1000, ttl_days: int = 30, enabled: bool = True):
        """
        Initialize the response cache.

        Args:
            max_entries: Maximum number of cache entries to keep
            ttl_days: Days before unused entries expire
            enabled: Whether caching is enabled
        """
        self.max_entries = max_entries
        self.ttl_days = ttl_days
        self.enabled = enabled
        self._hits = 0
        self._misses = 0

        if self.enabled:
            self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(CACHE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                scenario_id TEXT,
                query TEXT,
                answer_text TEXT,
                passed INTEGER,
                tier TEXT,
                created_at TEXT,
                accessed_at TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _make_key(self, query: str, chunk_ids: list[str]) -> str:
        """Create cache key from query and retrieved chunk IDs."""
        content = query + "|" + "|".join(sorted(chunk_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, query: str, chunk_ids: list[str]) -> Optional[dict]:
        """
        Check cache for existing validated response.

        Args:
            query: The query string
            chunk_ids: List of retrieved chunk IDs

        Returns:
            Dict with answer_text, passed, tier if found, None otherwise
        """
        if not self.enabled:
            return None

        key = self._make_key(query, chunk_ids)
        conn = sqlite3.connect(CACHE_DB)
        cur = conn.execute(
            "SELECT answer_text, passed, tier, scenario_id FROM cache WHERE cache_key = ?",
            (key,),
        )
        row = cur.fetchone()

        if row:
            # Update access time
            conn.execute(
                "UPDATE cache SET accessed_at = ? WHERE cache_key = ?",
                (datetime.utcnow().isoformat(), key),
            )
            conn.commit()
            self._hits += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
        else:
            self._misses += 1

        conn.close()

        if row:
            return {
                "answer_text": row[0],
                "passed": bool(row[1]),
                "tier": row[2],
                "scenario_id": row[3],
            }
        return None

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
        Store validated response in cache.

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

        key = self._make_key(query, chunk_ids)
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(CACHE_DB)
        conn.execute(
            """
            INSERT OR REPLACE INTO cache
            (cache_key, scenario_id, query, answer_text, passed, tier, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, scenario_id, query, answer_text, int(passed), tier, now, now),
        )
        conn.commit()

        # Periodic cleanup
        self._cleanup(conn)
        conn.close()

        logger.debug(f"Cached result for scenario {scenario_id} (tier={tier}, passed={passed})")

    def _cleanup(self, conn: sqlite3.Connection) -> None:
        """Remove old and excess entries."""
        # Remove expired entries
        cutoff = (datetime.utcnow() - timedelta(days=self.ttl_days)).isoformat()
        conn.execute("DELETE FROM cache WHERE accessed_at < ?", (cutoff,))

        # Keep only max_entries most recently accessed
        conn.execute(
            """
            DELETE FROM cache WHERE cache_key NOT IN (
                SELECT cache_key FROM cache ORDER BY accessed_at DESC LIMIT ?
            )
            """,
            (self.max_entries,),
        )
        conn.commit()

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries deleted
        """
        if not self.enabled or not CACHE_DB.exists():
            return 0

        conn = sqlite3.connect(CACHE_DB)
        cur = conn.execute("SELECT COUNT(*) FROM cache")
        count = cur.fetchone()[0]
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()

        logger.info(f"Cleared {count} cache entries")
        return count

    def stats(self) -> dict:
        """Get cache statistics."""
        if not self.enabled or not CACHE_DB.exists():
            return {"enabled": False, "entries": 0, "hits": 0, "misses": 0}

        conn = sqlite3.connect(CACHE_DB)
        cur = conn.execute("SELECT COUNT(*) FROM cache")
        count = cur.fetchone()[0]
        conn.close()

        hit_rate = (self._hits / (self._hits + self._misses) * 100) if (self._hits + self._misses) > 0 else 0

        return {
            "enabled": True,
            "entries": count,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }
