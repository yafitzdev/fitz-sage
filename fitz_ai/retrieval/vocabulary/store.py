# fitz_ai/retrieval/vocabulary/store.py
"""
Vocabulary store for persisting keywords to PostgreSQL.

The vocabulary is stored per-collection in the PostgreSQL database:
- Auto-detected keywords with variations
- User-defined keywords
- Metadata about detection

User modifications are preserved across re-ingests.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import STORAGE
from fitz_ai.storage import get_connection_manager

from .models import Keyword, VocabularyMetadata

logger = get_logger(__name__)


class VocabularyStore:
    """
    Manages keyword vocabulary persistence in PostgreSQL.

    Vocabulary is stored per-collection in the database.
    This ensures keywords from different collections don't mix.

    Usage:
        store = VocabularyStore(collection="my_collection")

        # Save keywords
        store.save(keywords, metadata)

        # Load keywords
        keywords = store.load()

        # Merge new detections with existing (preserves user edits)
        store.merge_and_save(new_keywords, source_docs=50)
    """

    SCHEMA_SQL = """
        -- Keywords table
        CREATE TABLE IF NOT EXISTS keywords (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            match TEXT[] NOT NULL DEFAULT '{}',
            occurrences INTEGER NOT NULL DEFAULT 1,
            first_seen TEXT,
            user_defined BOOLEAN NOT NULL DEFAULT FALSE,
            auto_generated TEXT[] NOT NULL DEFAULT '{}'
        );

        -- Vocabulary metadata (singleton)
        CREATE TABLE IF NOT EXISTS vocabulary_meta (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            generated TIMESTAMPTZ NOT NULL,
            source_docs INTEGER NOT NULL DEFAULT 0,
            auto_detected INTEGER NOT NULL DEFAULT 0,
            user_modified INTEGER NOT NULL DEFAULT 0
        );

        -- Index for category lookups
        CREATE INDEX IF NOT EXISTS idx_keywords_category
        ON keywords(category);
    """

    def __init__(self, collection: str | None = None, path=None):
        """
        Initialize the store.

        Args:
            collection: Collection name for per-collection vocabulary
            path: Ignored (kept for backwards compatibility)
        """
        self.collection = collection or "default"
        self._manager = get_connection_manager()
        self._manager.start()
        self._schema_initialized = False

    def _ensure_schema(self) -> None:
        """Create tables schema if not exists."""
        if self._schema_initialized:
            return

        with self._manager.connection(self.collection) as conn:
            conn.execute(self.SCHEMA_SQL)
            conn.commit()

        self._schema_initialized = True
        logger.debug(f"{STORAGE} Vocabulary schema initialized for '{self.collection}'")

    def exists(self) -> bool:
        """Check if vocabulary has any keywords."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute("SELECT COUNT(*) FROM keywords").fetchone()
            return result[0] > 0 if result else False

    def load(self) -> list[Keyword]:
        """
        Load keywords from PostgreSQL.

        Returns:
            List of keywords (empty if none exist)
        """
        self._ensure_schema()

        try:
            with self._manager.connection(self.collection) as conn:
                cursor = conn.execute(
                    """
                    SELECT id, category, match, occurrences, first_seen,
                           user_defined, auto_generated
                    FROM keywords
                    ORDER BY category, id
                    """
                )

                keywords = []
                for row in cursor.fetchall():
                    keywords.append(
                        Keyword(
                            id=row[0],
                            category=row[1],
                            match=list(row[2]) if row[2] else [],
                            occurrences=row[3],
                            first_seen=row[4],
                            user_defined=row[5],
                            auto_generated=list(row[6]) if row[6] else [],
                        )
                    )

                logger.debug(
                    f"[VOCABULARY] Loaded {len(keywords)} keywords for '{self.collection}'"
                )
                return keywords

        except Exception as e:
            logger.warning(f"[VOCABULARY] Failed to load keywords: {e}")
            return []

    def load_with_metadata(self) -> tuple[list[Keyword], VocabularyMetadata | None]:
        """
        Load keywords and metadata from PostgreSQL.

        Returns:
            Tuple of (keywords, metadata)
        """
        self._ensure_schema()

        try:
            keywords = self.load()

            with self._manager.connection(self.collection) as conn:
                result = conn.execute(
                    """
                    SELECT generated, source_docs, auto_detected, user_modified
                    FROM vocabulary_meta
                    WHERE id = 1
                    """
                ).fetchone()

                if result:
                    metadata = VocabularyMetadata(
                        generated=result[0] if result[0] else datetime.now(timezone.utc),
                        source_docs=result[1],
                        auto_detected=result[2],
                        user_modified=result[3],
                    )
                else:
                    metadata = None

            return keywords, metadata

        except Exception as e:
            logger.warning(f"[VOCABULARY] Failed to load keywords: {e}")
            return [], None

    def save(
        self,
        keywords: list[Keyword],
        metadata: VocabularyMetadata | None = None,
    ) -> None:
        """
        Save keywords to PostgreSQL.

        Args:
            keywords: Keywords to save
            metadata: Optional metadata
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            # Clear existing keywords and insert new ones
            conn.execute("DELETE FROM keywords")

            for kw in keywords:
                conn.execute(
                    """
                    INSERT INTO keywords (id, category, match, occurrences, first_seen,
                                         user_defined, auto_generated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        kw.id,
                        kw.category,
                        kw.match,
                        kw.occurrences,
                        kw.first_seen,
                        kw.user_defined,
                        kw.auto_generated,
                    ),
                )

            # Calculate metadata if not provided
            if not metadata:
                metadata = VocabularyMetadata(
                    auto_detected=len([k for k in keywords if not k.user_defined]),
                    user_modified=len([k for k in keywords if k.user_defined]),
                )

            # Upsert metadata
            conn.execute(
                """
                INSERT INTO vocabulary_meta (id, generated, source_docs, auto_detected, user_modified)
                VALUES (1, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    generated = EXCLUDED.generated,
                    source_docs = EXCLUDED.source_docs,
                    auto_detected = EXCLUDED.auto_detected,
                    user_modified = EXCLUDED.user_modified
                """,
                (
                    metadata.generated,
                    metadata.source_docs,
                    metadata.auto_detected,
                    metadata.user_modified,
                ),
            )

            conn.commit()

        logger.info(f"[VOCABULARY] Saved {len(keywords)} keywords for '{self.collection}'")

    def merge_and_save(
        self,
        new_keywords: list[Keyword],
        source_docs: int = 0,
    ) -> list[Keyword]:
        """
        Merge new detections with existing keywords, preserving user edits.

        User-added variations and user-defined keywords are preserved.

        Args:
            new_keywords: Newly detected keywords
            source_docs: Number of source documents scanned

        Returns:
            Merged list of keywords
        """
        existing, _ = self.load_with_metadata()

        # Build lookup of existing keywords by ID (case-insensitive)
        existing_by_id: dict[str, Keyword] = {kw.id.lower(): kw for kw in existing}

        merged: list[Keyword] = []

        # Process new keywords
        for new_kw in new_keywords:
            key = new_kw.id.lower()

            if key in existing_by_id:
                # Merge with existing
                old_kw = existing_by_id[key]

                # Preserve user-added variations
                user_variations = set(old_kw.match) - set(old_kw.auto_generated)

                # Combine: new auto-generated + user-added
                all_variations = set(new_kw.match) | user_variations
                new_kw.match = sorted(all_variations, key=str.lower)
                new_kw.auto_generated = new_kw.match.copy()

                # Preserve user_defined flag
                new_kw.user_defined = old_kw.user_defined

                # Update occurrences to max
                new_kw.occurrences = max(new_kw.occurrences, old_kw.occurrences)

                # Remove from existing (we've processed it)
                del existing_by_id[key]

            merged.append(new_kw)

        # Add remaining user-defined keywords that weren't re-detected
        for old_kw in existing_by_id.values():
            if old_kw.user_defined:
                merged.append(old_kw)

        # Calculate metadata
        auto_detected = len([k for k in merged if not k.user_defined])
        user_modified = len([k for k in merged if k.user_defined])

        metadata = VocabularyMetadata(
            generated=datetime.now(timezone.utc),
            source_docs=source_docs,
            auto_detected=auto_detected,
            user_modified=user_modified,
        )

        # Save
        self.save(merged, metadata)

        return merged

    def add_keyword(self, keyword: Keyword) -> None:
        """
        Add a single keyword to the vocabulary.

        Args:
            keyword: Keyword to add
        """
        self._ensure_schema()

        # Check if already exists
        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "SELECT id FROM keywords WHERE LOWER(id) = LOWER(%s)",
                (keyword.id,),
            ).fetchone()

            if result:
                logger.warning(f"[VOCABULARY] Keyword {keyword.id!r} already exists")
                return

            keyword.user_defined = True
            conn.execute(
                """
                INSERT INTO keywords (id, category, match, occurrences, first_seen,
                                     user_defined, auto_generated)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    keyword.id,
                    keyword.category,
                    keyword.match,
                    keyword.occurrences,
                    keyword.first_seen,
                    keyword.user_defined,
                    keyword.auto_generated,
                ),
            )
            conn.commit()

        logger.info(f"[VOCABULARY] Added keyword {keyword.id!r}")

    def add_variation(self, keyword_id: str, variation: str) -> bool:
        """
        Add a variation to an existing keyword.

        Args:
            keyword_id: ID of the keyword
            variation: Variation to add

        Returns:
            True if successful, False if keyword not found
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            # Get current match array
            result = conn.execute(
                "SELECT match FROM keywords WHERE LOWER(id) = LOWER(%s)",
                (keyword_id,),
            ).fetchone()

            if not result:
                logger.warning(f"[VOCABULARY] Keyword {keyword_id!r} not found")
                return False

            current_match = list(result[0]) if result[0] else []

            if variation not in current_match:
                current_match.append(variation)
                conn.execute(
                    "UPDATE keywords SET match = %s WHERE LOWER(id) = LOWER(%s)",
                    (current_match, keyword_id),
                )
                conn.commit()
                logger.info(f"[VOCABULARY] Added variation {variation!r} to {keyword_id}")

            return True

    def remove_keyword(self, keyword_id: str) -> bool:
        """
        Remove a keyword from the vocabulary.

        Args:
            keyword_id: ID of the keyword to remove

        Returns:
            True if removed, False if not found
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            result = conn.execute(
                "DELETE FROM keywords WHERE LOWER(id) = LOWER(%s) RETURNING id",
                (keyword_id,),
            ).fetchone()

            if result:
                conn.commit()
                logger.info(f"[VOCABULARY] Removed keyword {keyword_id!r}")
                return True

        logger.warning(f"[VOCABULARY] Keyword {keyword_id!r} not found")
        return False

    def get_by_category(self, category: str) -> list[Keyword]:
        """
        Get keywords by category.

        Args:
            category: Category to filter by

        Returns:
            Keywords in that category
        """
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute(
                """
                SELECT id, category, match, occurrences, first_seen,
                       user_defined, auto_generated
                FROM keywords
                WHERE category = %s
                ORDER BY id
                """,
                (category,),
            )

            return [
                Keyword(
                    id=row[0],
                    category=row[1],
                    match=list(row[2]) if row[2] else [],
                    occurrences=row[3],
                    first_seen=row[4],
                    user_defined=row[5],
                    auto_generated=list(row[6]) if row[6] else [],
                )
                for row in cursor.fetchall()
            ]

    def get_categories(self) -> list[str]:
        """Get all unique categories in the vocabulary."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            cursor = conn.execute("SELECT DISTINCT category FROM keywords ORDER BY category")
            return [row[0] for row in cursor.fetchall()]

    def clear(self) -> None:
        """Clear all keywords from the vocabulary."""
        self._ensure_schema()

        with self._manager.connection(self.collection) as conn:
            conn.execute("DELETE FROM keywords")
            conn.execute("DELETE FROM vocabulary_meta")
            conn.commit()

        logger.info(f"[VOCABULARY] Cleared vocabulary for '{self.collection}'")
