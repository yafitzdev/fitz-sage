# fitz_ai/ingestion/enrichment/entities/extractor.py
"""
Entity extractor for identifying entities in content.

The extractor:
1. Takes chunk content and file path
2. Builds a prompt based on content type
3. Extracts entities using an LLM
4. Caches results to avoid redundant API calls
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from fitz_ai.prompts.entities import EXTRACTION_PROMPT, EXTRACTION_PROMPT_CODE

from .cache import EntityCache
from .models import Entity

logger = logging.getLogger(__name__)

# File extensions that should use code-specific prompt
CODE_EXTENSIONS = frozenset(
    {
        ".py",
        ".pyw",  # Python
        ".js",
        ".jsx",
        ".ts",
        ".tsx",  # JavaScript/TypeScript
        ".java",
        ".kt",  # JVM
        ".go",  # Go
        ".rs",  # Rust
        ".c",
        ".cpp",
        ".h",
        ".hpp",  # C/C++
        ".rb",  # Ruby
        ".php",  # PHP
        ".cs",  # C#
        ".swift",  # Swift
    }
)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class EntityExtractor:
    """
    Extracts entities from content using LLM.

    Uses prompts tailored to content type (code vs document) and
    caches results to avoid redundant API calls.

    Usage:
        extractor = EntityExtractor(
            chat_client=my_llm_client,
            cache=EntityCache(cache_path),
            extractor_id="llm:gpt-4o-mini:v1",
        )

        entities = extractor.extract(
            content="class UserAuth: ...",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )
    """

    def __init__(
        self,
        *,
        chat_client: ChatClient,
        cache: EntityCache | None = None,
        extractor_id: str,
        entity_types: list[str] | None = None,
    ):
        self._chat = chat_client
        self._cache = cache
        self.extractor_id = extractor_id
        self._entity_types = set(entity_types) if entity_types else None

    def extract(
        self,
        content: str,
        file_path: str,
        content_hash: str,
    ) -> list[Entity]:
        """
        Extract entities from content.

        Args:
            content: The chunk content to analyze
            file_path: Path to the source file
            content_hash: Hash of the content (for caching)

        Returns:
            List of extracted entities
        """
        # Check cache first
        if self._cache is not None:
            cached = self._cache.get(content_hash, self.extractor_id)
            if cached is not None:
                logger.debug(f"Cache hit for entity extraction: {file_path}")
                return self._filter_entities(cached)

        # Build and execute prompt
        prompt = self._build_prompt(content, file_path)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._chat.chat(messages)
            entities = self._parse_response(response)
        except Exception as e:
            logger.warning(f"Entity extraction failed for {file_path}: {e}")
            entities = []

        # Cache result
        if self._cache is not None:
            self._cache.set(content_hash, self.extractor_id, entities)

        logger.debug(f"Extracted {len(entities)} entities from {file_path}")
        return self._filter_entities(entities)

    def _build_prompt(self, content: str, file_path: str) -> str:
        """Build the extraction prompt based on file type."""
        ext = Path(file_path).suffix.lower()

        # Truncate content to avoid token limits
        max_chars = 3000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"

        if ext in CODE_EXTENSIONS:
            return EXTRACTION_PROMPT_CODE.format(content=content)
        else:
            return EXTRACTION_PROMPT.format(content=content)

    def _parse_response(self, response: str) -> list[Entity]:
        """Parse LLM response into Entity objects."""
        # Try to extract JSON from response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines (```json and ```)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
            if not isinstance(data, list):
                logger.warning("Entity extraction response is not a list")
                return []

            entities = []
            for item in data:
                if isinstance(item, dict) and "name" in item and "type" in item:
                    entities.append(
                        Entity(
                            name=str(item["name"]),
                            type=str(item["type"]),
                            description=str(item.get("description", "")) or None,
                        )
                    )
            return entities

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction response: {e}")
            return []

    def _filter_entities(self, entities: list[Entity]) -> list[Entity]:
        """Filter entities by configured types."""
        if self._entity_types is None:
            return entities
        return [e for e in entities if e.type in self._entity_types]

    def save_cache(self) -> None:
        """Explicitly save the cache to disk."""
        if self._cache is not None:
            self._cache.save()


__all__ = ["EntityExtractor", "ChatClient"]
