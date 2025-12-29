# fitz_ai/engines/graphrag/graph/extraction.py
"""
Entity and Relationship Extraction for GraphRAG.

Uses LLM to extract entities and relationships from text chunks.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fitz_ai.engines.graphrag.config.schema import GraphExtractionConfig
from fitz_ai.engines.graphrag.graph.storage import Entity, KnowledgeGraph, Relationship
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

## Entity Types to Extract
{entity_types}

## Instructions
1. Identify all important entities (people, organizations, locations, concepts, etc.)
2. For each entity, provide: name, type, and a brief description
3. Identify relationships between entities
4. For each relationship, provide: source entity, target entity, relationship type, and description

## Output Format
Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "entity_type", "description": "Brief description"}}
  ],
  "relationships": [
    {{"source": "Source Entity Name", "target": "Target Entity Name", "type": "relationship_type", "description": "Brief description"}}
  ]
}}

## Text to Analyze
{text}

## Output (JSON only, no other text):"""


class EntityRelationshipExtractor:
    """
    Extracts entities and relationships from text using LLM.

    This is the core extraction component of GraphRAG that builds
    the knowledge graph from document chunks.

    Uses tier="fast" for extraction since this is a background task
    processing many chunks.
    """

    def __init__(
        self,
        config: GraphExtractionConfig,
        llm_provider: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration
            llm_provider: LLM provider name (uses default from config if None)
        """
        self.config = config
        self._llm_provider = llm_provider or "cohere"
        self._chat_client = None

    def _get_chat_client(self):
        """Lazy load chat client with 'fast' tier for background processing."""
        if self._chat_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._chat_client = get_llm_plugin(
                plugin_type="chat",
                plugin_name=self._llm_provider,
                tier="fast",  # Use fast model for background extraction
            )
        return self._chat_client

    def extract_from_text(
        self, text: str, chunk_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from a single text chunk.

        Args:
            text: Text to extract from
            chunk_id: Optional chunk ID for provenance

        Returns:
            Tuple of (entities, relationships)
        """
        # Build prompt
        entity_types_str = (
            ", ".join(self.config.entity_types) if self.config.entity_types else "any"
        )
        prompt = EXTRACTION_PROMPT.format(
            entity_types=entity_types_str,
            text=text[:8000],  # Truncate very long texts
        )

        # Call LLM
        chat_client = self._get_chat_client()
        response = chat_client.chat([{"role": "user", "content": prompt}])

        # Parse response
        entities, relationships = self._parse_extraction_response(response, chunk_id)

        # Apply limits
        entities = entities[: self.config.max_entities_per_chunk]
        relationships = relationships[: self.config.max_relationships_per_chunk]

        return entities, relationships

    def _parse_extraction_response(
        self, response: str, chunk_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Parse LLM response into entities and relationships.

        Args:
            response: LLM response text
            chunk_id: Optional chunk ID for provenance

        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("No JSON found in extraction response")
            return entities, relationships

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}")
            return entities, relationships

        # Parse entities
        entity_name_to_id: Dict[str, str] = {}
        for e_data in data.get("entities", []):
            if not isinstance(e_data, dict):
                continue

            name = e_data.get("name", "").strip()
            if not name:
                continue

            entity_id = str(uuid.uuid4())[:8]
            entity_name_to_id[name.lower()] = entity_id

            entity = Entity(
                id=entity_id,
                name=name,
                type=e_data.get("type", "unknown").lower(),
                description=e_data.get("description", ""),
                source_chunks=[chunk_id] if chunk_id else [],
            )
            entities.append(entity)

        # Parse relationships
        for r_data in data.get("relationships", []):
            if not isinstance(r_data, dict):
                continue

            source_name = r_data.get("source", "").strip().lower()
            target_name = r_data.get("target", "").strip().lower()

            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if not source_id or not target_id:
                continue

            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                type=r_data.get("type", "related_to").lower().replace(" ", "_"),
                description=r_data.get("description", ""),
                source_chunks=[chunk_id] if chunk_id else [],
            )
            relationships.append(rel)

        return entities, relationships

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        graph: Optional[KnowledgeGraph] = None,
    ) -> KnowledgeGraph:
        """
        Extract entities and relationships from multiple chunks.

        Args:
            chunks: List of chunk dicts with 'text' and optionally 'id' keys
            graph: Optional existing graph to add to

        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        if graph is None:
            graph = KnowledgeGraph()

        # Track entity names to merge duplicates
        name_to_entity: Dict[str, Entity] = {}

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("id", str(uuid.uuid4())[:8])

            if not text.strip():
                continue

            try:
                entities, relationships = self.extract_from_text(text, chunk_id)
            except Exception as e:
                logger.warning(f"Extraction failed for chunk {chunk_id}: {e}")
                continue

            # Merge entities by name
            for entity in entities:
                name_key = entity.name.lower()
                if name_key in name_to_entity:
                    # Merge source chunks
                    existing = name_to_entity[name_key]
                    existing.source_chunks.extend(entity.source_chunks)
                    # Update description if new one is longer
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description
                else:
                    name_to_entity[name_key] = entity
                    graph.add_entity(entity)

            # Add relationships (map to merged entity IDs)
            for rel in relationships:
                # Find actual entity IDs after merging
                source_entity = None
                target_entity = None

                for entity in entities:
                    if entity.id == rel.source_id:
                        source_entity = name_to_entity.get(entity.name.lower())
                    if entity.id == rel.target_id:
                        target_entity = name_to_entity.get(entity.name.lower())

                if source_entity and target_entity:
                    rel.source_id = source_entity.id
                    rel.target_id = target_entity.id
                    graph.add_relationship(rel)

        logger.info(
            f"Extracted {graph.num_entities} entities and "
            f"{graph.num_relationships} relationships from {len(chunks)} chunks"
        )

        return graph
