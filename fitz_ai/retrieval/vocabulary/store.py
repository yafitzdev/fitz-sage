# fitz_ai/retrieval/vocabulary/store.py
"""
Vocabulary store for persisting keywords to YAML.

The vocabulary is stored in .fitz/keywords.yaml and contains:
- Auto-detected keywords with variations
- User-defined keywords
- Metadata about detection

User modifications are preserved across re-ingests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

from .models import Keyword, VocabularyMetadata

logger = get_logger(__name__)


class VocabularyStore:
    """
    Manages keyword vocabulary persistence.

    Vocabulary is stored per-collection in .fitz/keywords/{collection}.yaml.
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

    def __init__(self, collection: str | None = None, path: Path | None = None):
        """
        Initialize the store.

        Args:
            collection: Collection name for per-collection vocabulary
            path: Override path (defaults to .fitz/keywords/{collection}.yaml)
        """
        self.collection = collection
        if path:
            self.path = path
        else:
            self.path = FitzPaths.vocabulary(collection)

    def exists(self) -> bool:
        """Check if vocabulary file exists."""
        return self.path.exists()

    def load(self) -> list[Keyword]:
        """
        Load keywords from YAML file.

        Returns:
            List of keywords (empty if file doesn't exist)
        """
        if not self.path.exists():
            return []

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "keywords" not in data:
                return []

            keywords = [Keyword.from_dict(kw) for kw in data["keywords"]]
            logger.debug(f"[VOCABULARY] Loaded {len(keywords)} keywords from {self.path}")
            return keywords

        except Exception as e:
            logger.warning(f"[VOCABULARY] Failed to load keywords: {e}")
            return []

    def load_with_metadata(self) -> tuple[list[Keyword], VocabularyMetadata | None]:
        """
        Load keywords and metadata from YAML file.

        Returns:
            Tuple of (keywords, metadata)
        """
        if not self.path.exists():
            return [], None

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                return [], None

            keywords = [Keyword.from_dict(kw) for kw in data.get("keywords", [])]
            metadata = None
            if "_meta" in data:
                metadata = VocabularyMetadata.from_dict(data["_meta"])

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
        Save keywords to YAML file.

        Args:
            keywords: Keywords to save
            metadata: Optional metadata
        """
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {}

        # Add metadata
        if metadata:
            data["_meta"] = metadata.to_dict()
        else:
            data["_meta"] = VocabularyMetadata(
                auto_detected=len([k for k in keywords if not k.user_defined]),
                user_modified=len([k for k in keywords if k.user_defined]),
            ).to_dict()

        # Add keywords grouped by category
        data["keywords"] = [kw.to_dict() for kw in keywords]

        # Write with nice formatting
        with open(self.path, "w", encoding="utf-8") as f:
            # Add header comment
            f.write("# Auto-generated vocabulary file\n")
            f.write("# Edit freely - your changes are preserved on re-ingest\n")
            f.write("#\n")
            f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("#\n\n")

            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(f"[VOCABULARY] Saved {len(keywords)} keywords to {self.path}")

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
        existing, existing_meta = self.load_with_metadata()

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
        keywords = self.load()

        # Check if already exists
        existing_ids = {kw.id.lower() for kw in keywords}
        if keyword.id.lower() in existing_ids:
            logger.warning(f"[VOCABULARY] Keyword {keyword.id!r} already exists")
            return

        keyword.user_defined = True
        keywords.append(keyword)
        self.save(keywords)

    def add_variation(self, keyword_id: str, variation: str) -> bool:
        """
        Add a variation to an existing keyword.

        Args:
            keyword_id: ID of the keyword
            variation: Variation to add

        Returns:
            True if successful, False if keyword not found
        """
        keywords = self.load()

        for kw in keywords:
            if kw.id.lower() == keyword_id.lower():
                if variation not in kw.match:
                    kw.match.append(variation)
                    self.save(keywords)
                    logger.info(f"[VOCABULARY] Added variation {variation!r} to {keyword_id}")
                return True

        logger.warning(f"[VOCABULARY] Keyword {keyword_id!r} not found")
        return False

    def remove_keyword(self, keyword_id: str) -> bool:
        """
        Remove a keyword from the vocabulary.

        Args:
            keyword_id: ID of the keyword to remove

        Returns:
            True if removed, False if not found
        """
        keywords = self.load()
        original_count = len(keywords)

        keywords = [kw for kw in keywords if kw.id.lower() != keyword_id.lower()]

        if len(keywords) < original_count:
            self.save(keywords)
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
        keywords = self.load()
        return [kw for kw in keywords if kw.category == category]

    def get_categories(self) -> list[str]:
        """Get all unique categories in the vocabulary."""
        keywords = self.load()
        return sorted(set(kw.category for kw in keywords))
