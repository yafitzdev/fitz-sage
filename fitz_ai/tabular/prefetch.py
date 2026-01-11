# fitz_ai/tabular/prefetch.py
"""Background prefetcher for team mode table data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_ai.tabular.store.qdrant import QdrantTableStore

logger = get_logger(__name__)


class TablePrefetcher:
    """
    Prefetch table data in background for team mode.

    When a collection is loaded, starts parallel prefetch of all tables
    from Qdrant to local cache. By the time the first query arrives,
    the cache is warm - no perceived cold start.

    Uses parallel loading (2 workers by default) for 2x speedup.
    """

    def __init__(self, store: "QdrantTableStore", max_workers: int = 2):
        self.store = store
        self.max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list = []
        self._started = False
        self._completed = False

    def start(self) -> None:
        """
        Start background prefetch of all tables.

        This method returns immediately. Prefetch runs in background threads.
        Call wait() if you need to block until prefetch completes.
        """
        if self._started:
            return

        table_ids = self.store.list_tables()
        if not table_ids:
            self._completed = True
            return

        logger.info(f"Starting background prefetch of {len(table_ids)} tables...")
        self._started = True

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="table_prefetch",
        )

        self._futures = [
            self._executor.submit(self._prefetch_one, tid) for tid in table_ids
        ]

    def _prefetch_one(self, table_id: str) -> bool:
        """
        Prefetch single table (runs in background thread).

        Returns True if table was fetched successfully.
        """
        try:
            # This will check cache, fetch from Qdrant if needed
            result = self.store.retrieve(table_id)
            if result:
                logger.debug(f"Prefetched table {table_id}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to prefetch table {table_id}: {e}")
            return False

    def wait(self, timeout: float | None = None) -> int:
        """
        Wait for prefetch to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            Number of tables successfully prefetched
        """
        if not self._started or self._completed:
            return 0

        if not self._futures:
            return 0

        success_count = 0
        for future in as_completed(self._futures, timeout=timeout):
            try:
                if future.result():
                    success_count += 1
            except Exception:
                pass

        self._completed = True
        logger.info(f"Prefetch complete: {success_count} tables cached")
        return success_count

    def is_complete(self) -> bool:
        """Check if prefetch has completed."""
        if self._completed:
            return True

        if not self._started or not self._futures:
            return True

        return all(f.done() for f in self._futures)

    def shutdown(self) -> None:
        """Shutdown executor (cancels pending prefetches)."""
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
