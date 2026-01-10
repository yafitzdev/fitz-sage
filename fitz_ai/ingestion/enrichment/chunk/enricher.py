# fitz_ai/ingestion/enrichment/chunk/enricher.py
"""
Chunk Enrichment Bus - Unified per-chunk enrichment with extensible modules.

The enrichment bus runs one LLM call per batch of chunks and extracts
multiple enrichments (summary, keywords, entities, etc.) in a single pass.

Architecture:
    ChunkEnricher
        ├── SummaryModule      → chunk.metadata["summary"]
        ├── KeywordModule      → VocabularyStore
        ├── EntityModule       → chunk.metadata["entities"]
        └── (future modules)   → easily extensible

Usage:
    enricher = ChunkEnricher(
        chat_client=fast_chat,
        modules=[SummaryModule(), KeywordModule(), EntityModule()],
    )
    enriched_chunks, keywords = enricher.enrich(chunks)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fitz_ai.core.chunk import Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


# =============================================================================
# Enrichment Module Interface
# =============================================================================


class EnrichmentModule(ABC):
    """
    Base class for enrichment modules.

    Each module defines:
    - What to extract (prompt_instruction)
    - How to parse the result (parse_result)
    - Where to store it (apply_to_chunk or collect separately)

    To add a new enrichment:
    1. Subclass EnrichmentModule
    2. Implement the abstract methods
    3. Add to ChunkEnricher's module list
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this module."""
        ...

    @property
    @abstractmethod
    def json_key(self) -> str:
        """Key in the JSON response for this module's output."""
        ...

    @abstractmethod
    def prompt_instruction(self) -> str:
        """
        Return the instruction to include in the prompt.

        Should describe what to extract and the expected format.
        """
        ...

    @abstractmethod
    def parse_result(self, data: Any) -> Any:
        """
        Parse and validate the module's output from the JSON response.

        Args:
            data: The value from response[json_key]

        Returns:
            Parsed/validated result
        """
        ...

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        """
        Apply the enrichment result to a chunk's metadata.

        Override this if the module should attach data to chunks.
        Default implementation does nothing (for modules like keywords
        that collect data separately).
        """
        pass


# =============================================================================
# Built-in Modules
# =============================================================================


class SummaryModule(EnrichmentModule):
    """Extracts a searchable summary for each chunk."""

    @property
    def name(self) -> str:
        return "summary"

    @property
    def json_key(self) -> str:
        return "summary"

    def prompt_instruction(self) -> str:
        return (
            '"summary": A 2-3 sentence description of what this content does/contains '
            "and when someone would search for it. Be specific, not vague."
        )

    def parse_result(self, data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        return str(data).strip() if data else ""

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        if result:
            chunk.metadata["summary"] = result


class KeywordModule(EnrichmentModule):
    """
    Extracts exact-match keywords/identifiers.

    Keywords are collected separately (not attached to chunks) and
    saved to VocabularyStore for exact-match retrieval.
    """

    @property
    def name(self) -> str:
        return "keywords"

    @property
    def json_key(self) -> str:
        return "keywords"

    def prompt_instruction(self) -> str:
        return (
            '"keywords": Array of exact identifiers someone might search for verbatim: '
            "IDs (TC-1234, JIRA-567), function/class names (getUserById, AuthService), "
            "constants (MAX_RETRIES, API_KEY), version numbers (v2.0.1), error codes, "
            "API endpoints, etc. Only specific searchable terms, not generic words. "
            "Return empty array if none found."
        )

    def parse_result(self, data: Any) -> list[str]:
        if isinstance(data, list):
            return [str(item).strip() for item in data if item]
        return []

    # Keywords don't attach to chunks - they're collected separately


class EntityModule(EnrichmentModule):
    """Extracts named entities (people, organizations, concepts, etc.)."""

    @property
    def name(self) -> str:
        return "entities"

    @property
    def json_key(self) -> str:
        return "entities"

    def prompt_instruction(self) -> str:
        return (
            '"entities": Array of named entities as objects with "name" and "type" fields. '
            'Types: person, organization, product, technology, concept. '
            'Example: [{"name": "John Smith", "type": "person"}, '
            '{"name": "PostgreSQL", "type": "technology"}]. '
            "Return empty array if none found."
        )

    def parse_result(self, data: Any) -> list[dict[str, str]]:
        if not isinstance(data, list):
            return []

        entities = []
        for item in data:
            if isinstance(item, dict) and "name" in item:
                entities.append(
                    {
                        "name": str(item.get("name", "")),
                        "type": str(item.get("type", "unknown")),
                    }
                )
        return entities

    def apply_to_chunk(self, chunk: "Chunk", result: Any) -> None:
        if result:
            chunk.metadata["entities"] = result


# =============================================================================
# Enrichment Result
# =============================================================================


@dataclass
class ChunkEnrichmentResult:
    """Result of enriching a single chunk."""

    chunk_index: int
    results: dict[str, Any] = field(default_factory=dict)

    def get(self, module_name: str) -> Any:
        """Get result for a specific module."""
        return self.results.get(module_name)


@dataclass
class EnrichmentBatchResult:
    """Result of enriching a batch of chunks."""

    chunks: list["Chunk"]
    all_keywords: list[str] = field(default_factory=list)


# =============================================================================
# Chunk Enricher (The Bus)
# =============================================================================


@dataclass
class ChunkEnricher:
    """
    Unified chunk enrichment bus.

    Runs one LLM call per batch of chunks, extracting all configured
    enrichments in a single pass. Modules are pluggable - add new
    enrichment types by implementing EnrichmentModule.

    Usage:
        enricher = ChunkEnricher(
            chat_client=fast_chat,
            modules=[SummaryModule(), KeywordModule(), EntityModule()],
        )
        result = enricher.enrich(chunks)
        # result.chunks have metadata attached
        # result.all_keywords collected for VocabularyStore
    """

    chat_client: ChatClient
    modules: list[EnrichmentModule] = field(default_factory=list)
    batch_size: int = 15  # Chunks per LLM call
    max_content_length: int = 1500  # Truncate long chunks

    def __post_init__(self) -> None:
        if not self.modules:
            # Default modules if none specified
            self.modules = [SummaryModule(), KeywordModule(), EntityModule()]

    def enrich(self, chunks: list["Chunk"]) -> EnrichmentBatchResult:
        """
        Enrich all chunks with configured modules.

        Args:
            chunks: List of chunks to enrich

        Returns:
            EnrichmentBatchResult with enriched chunks and collected keywords
        """
        if not chunks:
            return EnrichmentBatchResult(chunks=[])

        all_keywords: list[str] = []

        # Process in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"[ENRICH] Batch {batch_num}/{total_batches}: "
                f"enriching {len(batch)} chunks"
            )

            # Run enrichment for this batch
            batch_results = self._enrich_batch(batch)

            # Apply results to chunks and collect keywords
            for chunk, result in zip(batch, batch_results):
                for module in self.modules:
                    module_result = result.get(module.name)
                    if module_result is not None:
                        module.apply_to_chunk(chunk, module_result)

                        # Collect keywords separately
                        if module.name == "keywords" and isinstance(module_result, list):
                            all_keywords.extend(module_result)

        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(all_keywords))

        logger.info(
            f"[ENRICH] Completed: {len(chunks)} chunks, "
            f"{len(unique_keywords)} unique keywords"
        )

        return EnrichmentBatchResult(chunks=chunks, all_keywords=unique_keywords)

    def _enrich_batch(self, batch: list["Chunk"]) -> list[ChunkEnrichmentResult]:
        """Enrich a single batch of chunks with one LLM call."""
        prompt = self._build_prompt(batch)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.chat_client.chat(messages)
            return self._parse_response(response, len(batch))
        except Exception as e:
            logger.error(f"[ENRICH] LLM call failed: {e}")
            # Return empty results on failure
            return [
                ChunkEnrichmentResult(chunk_index=i) for i in range(len(batch))
            ]

    def _build_prompt(self, batch: list["Chunk"]) -> str:
        """Build the combined prompt for a batch."""
        # Build module instructions
        module_instructions = ",\n  ".join(
            module.prompt_instruction() for module in self.modules
        )

        prompt_parts = [
            "Analyze each numbered chunk and extract structured information.\n\n"
            "For each chunk, return a JSON object with:\n"
            "{\n"
            f"  {module_instructions}\n"
            "}\n\n"
            "Return a JSON array with one object per chunk, in order.\n"
            "Example response format:\n"
            "```json\n"
            "[\n"
            '  {"summary": "...", "keywords": [...], "entities": [...]},\n'
            '  {"summary": "...", "keywords": [...], "entities": [...]}\n'
            "]\n"
            "```\n\n"
            "CHUNKS TO ANALYZE:\n"
        ]

        for i, chunk in enumerate(batch, 1):
            source = chunk.metadata.get("source_file", chunk.doc_id)
            file_name = Path(source).name if source else f"chunk_{i}"

            # Truncate content
            content = chunk.content
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "\n... [truncated]"

            prompt_parts.append(f"\n--- CHUNK [{i}] from {file_name} ---\n{content}\n")

        prompt_parts.append(
            "\n--- END OF CHUNKS ---\n\n"
            "Now return the JSON array with analysis for each chunk:"
        )

        return "".join(prompt_parts)

    def _parse_response(
        self, response: str, expected_count: int
    ) -> list[ChunkEnrichmentResult]:
        """Parse the LLM response into enrichment results."""
        results: list[ChunkEnrichmentResult] = []

        # Try to extract JSON from response
        json_data = self._extract_json(response)

        if isinstance(json_data, list):
            for i, item in enumerate(json_data):
                if i >= expected_count:
                    break
                result = ChunkEnrichmentResult(chunk_index=i)
                if isinstance(item, dict):
                    for module in self.modules:
                        if module.json_key in item:
                            parsed = module.parse_result(item[module.json_key])
                            result.results[module.name] = parsed
                results.append(result)

        # Fill in missing results
        while len(results) < expected_count:
            results.append(ChunkEnrichmentResult(chunk_index=len(results)))

        return results

    def _extract_json(self, response: str) -> Any:
        """Extract JSON from LLM response, handling markdown code blocks."""
        text = response.strip()

        # Try to find JSON in code block
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find array in text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("[ENRICH] Failed to parse JSON from response")
        return []


# =============================================================================
# Factory
# =============================================================================


def create_default_enricher(chat_client: ChatClient) -> ChunkEnricher:
    """Create an enricher with default modules."""
    return ChunkEnricher(
        chat_client=chat_client,
        modules=[
            SummaryModule(),
            KeywordModule(),
            EntityModule(),
        ],
    )


__all__ = [
    "EnrichmentModule",
    "SummaryModule",
    "KeywordModule",
    "EntityModule",
    "ChunkEnricher",
    "ChunkEnrichmentResult",
    "EnrichmentBatchResult",
    "create_default_enricher",
]
