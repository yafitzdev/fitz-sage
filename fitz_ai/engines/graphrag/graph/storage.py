# fitz_ai/engines/graphrag/graph/storage.py
"""
Knowledge Graph Storage using NetworkX.

This module provides a NetworkX-based knowledge graph for storing
entities, relationships, and community structure.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx


@dataclass
class Entity:
    """An entity in the knowledge graph."""

    id: str
    name: str
    type: str
    description: str = ""
    source_chunks: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """A relationship between entities."""

    source_id: str
    target_id: str
    type: str
    description: str = ""
    weight: float = 1.0
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class Community:
    """A community of related entities."""

    id: str
    level: int
    entity_ids: List[str]
    summary: str = ""
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)


class KnowledgeGraph:
    """
    NetworkX-based knowledge graph for GraphRAG.

    Stores entities as nodes and relationships as edges,
    with support for community detection and hierarchical structure.
    """

    def __init__(self):
        """Initialize empty knowledge graph."""
        self._graph = nx.Graph()
        self._entities: Dict[str, Entity] = {}
        self._relationships: List[Relationship] = []
        self._communities: Dict[str, Community] = {}
        self._entity_to_community: Dict[str, str] = {}

    @property
    def num_entities(self) -> int:
        """Number of entities in the graph."""
        return len(self._entities)

    @property
    def num_relationships(self) -> int:
        """Number of relationships in the graph."""
        return len(self._relationships)

    @property
    def num_communities(self) -> int:
        """Number of communities."""
        return len(self._communities)

    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.

        Args:
            entity: Entity to add
        """
        self._entities[entity.id] = entity
        self._graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
        )

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship to the graph.

        Args:
            relationship: Relationship to add
        """
        # Only add if both entities exist
        if relationship.source_id not in self._entities:
            return
        if relationship.target_id not in self._entities:
            return

        self._relationships.append(relationship)
        self._graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            type=relationship.type,
            description=relationship.description,
            weight=relationship.weight,
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self._entities.values() if e.type == entity_type]

    def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity."""
        return [
            r for r in self._relationships if r.source_id == entity_id or r.target_id == entity_id
        ]

    def get_neighbors(self, entity_id: str, max_hops: int = 1) -> Set[str]:
        """
        Get neighboring entities within max_hops.

        Args:
            entity_id: Starting entity
            max_hops: Maximum number of hops

        Returns:
            Set of neighboring entity IDs
        """
        if entity_id not in self._graph:
            return set()

        neighbors = set()
        current_level = {entity_id}

        for _ in range(max_hops):
            next_level = set()
            for node in current_level:
                for neighbor in self._graph.neighbors(node):
                    if neighbor not in neighbors and neighbor != entity_id:
                        neighbors.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level

        return neighbors

    def set_communities(self, communities: List[Community]) -> None:
        """
        Set community structure.

        Args:
            communities: List of communities
        """
        self._communities.clear()
        self._entity_to_community.clear()

        for community in communities:
            self._communities[community.id] = community
            for entity_id in community.entity_ids:
                self._entity_to_community[entity_id] = community.id

    def get_community(self, community_id: str) -> Optional[Community]:
        """Get community by ID."""
        return self._communities.get(community_id)

    def get_entity_community(self, entity_id: str) -> Optional[Community]:
        """Get the community an entity belongs to."""
        community_id = self._entity_to_community.get(entity_id)
        if community_id:
            return self._communities.get(community_id)
        return None

    def get_all_communities(self, level: Optional[int] = None) -> List[Community]:
        """
        Get all communities, optionally filtered by level.

        Args:
            level: Optional hierarchy level to filter by

        Returns:
            List of communities
        """
        if level is None:
            return list(self._communities.values())
        return [c for c in self._communities.values() if c.level == level]

    def search_entities(
        self, query: str, entity_types: Optional[List[str]] = None, limit: int = 10
    ) -> List[Entity]:
        """
        Search entities by name/description (simple substring match).

        Args:
            query: Search query
            entity_types: Optional list of entity types to filter
            limit: Maximum results

        Returns:
            Matching entities
        """
        query_lower = query.lower()
        results = []

        for entity in self._entities.values():
            if entity_types and entity.type not in entity_types:
                continue

            # Simple substring matching
            if query_lower in entity.name.lower():
                results.append(entity)
            elif query_lower in entity.description.lower():
                results.append(entity)

            if len(results) >= limit:
                break

        return results

    def get_subgraph(self, entity_ids: List[str]) -> "KnowledgeGraph":
        """
        Extract a subgraph containing only specified entities.

        Args:
            entity_ids: Entity IDs to include

        Returns:
            New KnowledgeGraph with subset of entities
        """
        subgraph = KnowledgeGraph()

        # Add entities
        for entity_id in entity_ids:
            entity = self._entities.get(entity_id)
            if entity:
                subgraph.add_entity(entity)

        # Add relationships between included entities
        entity_set = set(entity_ids)
        for rel in self._relationships:
            if rel.source_id in entity_set and rel.target_id in entity_set:
                subgraph.add_relationship(rel)

        return subgraph

    def to_context_string(
        self, entity_ids: Optional[List[str]] = None, include_relationships: bool = True
    ) -> str:
        """
        Convert graph (or subset) to a context string for LLM.

        Args:
            entity_ids: Optional list of entity IDs to include (None = all)
            include_relationships: Whether to include relationships

        Returns:
            Formatted string representation
        """
        lines = []

        # Get entities to include
        if entity_ids:
            entities = [self._entities[eid] for eid in entity_ids if eid in self._entities]
        else:
            entities = list(self._entities.values())

        # Format entities
        if entities:
            lines.append("## Entities\n")
            for entity in entities:
                lines.append(f"- **{entity.name}** ({entity.type})")
                if entity.description:
                    lines.append(f"  {entity.description}")

        # Format relationships
        if include_relationships:
            entity_set = {e.id for e in entities}
            relevant_rels = [
                r
                for r in self._relationships
                if r.source_id in entity_set and r.target_id in entity_set
            ]

            if relevant_rels:
                lines.append("\n## Relationships\n")
                for rel in relevant_rels:
                    source = self._entities.get(rel.source_id)
                    target = self._entities.get(rel.target_id)
                    if source and target:
                        lines.append(f"- {source.name} --[{rel.type}]--> {target.name}")
                        if rel.description:
                            lines.append(f"  {rel.description}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all data from the graph."""
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()
        self._communities.clear()
        self._entity_to_community.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "source_chunks": e.source_chunks,
                    "attributes": e.attributes,
                }
                for e in self._entities.values()
            ],
            "relationships": [
                {
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "type": r.type,
                    "description": r.description,
                    "weight": r.weight,
                    "source_chunks": r.source_chunks,
                }
                for r in self._relationships
            ],
            "communities": [
                {
                    "id": c.id,
                    "level": c.level,
                    "entity_ids": c.entity_ids,
                    "summary": c.summary,
                    "parent_id": c.parent_id,
                    "child_ids": c.child_ids,
                }
                for c in self._communities.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        """Deserialize graph from dictionary."""
        graph = cls()

        for e_data in data.get("entities", []):
            entity = Entity(
                id=e_data["id"],
                name=e_data["name"],
                type=e_data["type"],
                description=e_data.get("description", ""),
                source_chunks=e_data.get("source_chunks", []),
                attributes=e_data.get("attributes", {}),
            )
            graph.add_entity(entity)

        for r_data in data.get("relationships", []):
            rel = Relationship(
                source_id=r_data["source_id"],
                target_id=r_data["target_id"],
                type=r_data["type"],
                description=r_data.get("description", ""),
                weight=r_data.get("weight", 1.0),
                source_chunks=r_data.get("source_chunks", []),
            )
            graph.add_relationship(rel)

        communities = []
        for c_data in data.get("communities", []):
            community = Community(
                id=c_data["id"],
                level=c_data["level"],
                entity_ids=c_data["entity_ids"],
                summary=c_data.get("summary", ""),
                parent_id=c_data.get("parent_id"),
                child_ids=c_data.get("child_ids", []),
            )
            communities.append(community)
        graph.set_communities(communities)

        return graph

    def save(self, path: str) -> None:
        """
        Save graph to file.

        Args:
            path: File path to save to
        """
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """
        Load graph from file.

        Args:
            path: File path to load from

        Returns:
            Loaded KnowledgeGraph
        """
        import json

        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
