# fitz_ai/engines/graphrag/graph/community.py
"""
Community Detection and Summarization for GraphRAG.

Detects communities in the knowledge graph and generates
hierarchical summaries for global search.
"""

from typing import List, Optional

from fitz_ai.engines.graphrag.config.schema import GraphCommunityConfig
from fitz_ai.engines.graphrag.graph.storage import Community, KnowledgeGraph
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


COMMUNITY_SUMMARY_PROMPT = """Summarize the following group of related entities and their relationships.

## Entities in this Community
{entities}

## Relationships
{relationships}

## Instructions
Write a concise summary (2-4 sentences) that captures:
1. The main theme or topic of this group
2. Key entities and how they relate
3. Important facts or insights

## Summary:"""


class CommunityDetector:
    """
    Detects communities in the knowledge graph.

    Uses Louvain or Leiden algorithm to find clusters of
    related entities.
    """

    def __init__(self, config: GraphCommunityConfig):
        """
        Initialize detector.

        Args:
            config: Community detection configuration
        """
        self.config = config

    def detect_communities(self, graph: KnowledgeGraph, max_levels: int = 2) -> List[Community]:
        """
        Detect communities in the graph.

        Args:
            graph: Knowledge graph to analyze
            max_levels: Maximum hierarchy levels

        Returns:
            List of communities
        """
        if graph.num_entities == 0:
            return []

        # Use NetworkX community detection
        try:
            if self.config.algorithm == "leiden":
                communities = self._detect_leiden(graph)
            else:
                communities = self._detect_louvain(graph)
        except Exception as e:
            logger.warning(f"Community detection failed: {e}, using connected components")
            communities = self._detect_connected_components(graph)

        # Filter by minimum size
        communities = [
            c for c in communities if len(c.entity_ids) >= self.config.min_community_size
        ]

        # Build hierarchy if needed
        if max_levels > 1 and len(communities) > 1:
            communities = self._build_hierarchy(communities, max_levels)

        logger.info(f"Detected {len(communities)} communities")
        return communities

    def _detect_louvain(self, graph: KnowledgeGraph) -> List[Community]:
        """Detect communities using Louvain algorithm."""
        import networkx.algorithms.community as nx_comm

        # Get partition
        partition = nx_comm.louvain_communities(
            graph._graph,
            resolution=self.config.resolution,
            seed=42,
        )

        communities = []
        for i, entity_ids in enumerate(partition):
            community = Community(
                id=f"c_{i}",
                level=0,
                entity_ids=list(entity_ids),
            )
            communities.append(community)

        return communities

    def _detect_leiden(self, graph: KnowledgeGraph) -> List[Community]:
        """Detect communities using Leiden algorithm (requires leidenalg)."""
        try:
            import igraph as ig
            import leidenalg

            # Convert NetworkX to igraph
            edges = list(graph._graph.edges())
            if not edges:
                return self._detect_connected_components(graph)

            nodes = list(graph._graph.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}

            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(nodes))
            ig_graph.add_edges([(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges])

            # Run Leiden
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.ModularityVertexPartition,
                resolution_parameter=self.config.resolution,
                seed=42,
            )

            communities = []
            for i, cluster in enumerate(partition):
                entity_ids = [nodes[idx] for idx in cluster]
                community = Community(
                    id=f"c_{i}",
                    level=0,
                    entity_ids=entity_ids,
                )
                communities.append(community)

            return communities

        except ImportError:
            logger.warning("leidenalg not installed, falling back to Louvain")
            return self._detect_louvain(graph)

    def _detect_connected_components(self, graph: KnowledgeGraph) -> List[Community]:
        """Fallback: use connected components as communities."""
        import networkx as nx

        components = list(nx.connected_components(graph._graph))

        communities = []
        for i, entity_ids in enumerate(components):
            community = Community(
                id=f"c_{i}",
                level=0,
                entity_ids=list(entity_ids),
            )
            communities.append(community)

        return communities

    def _build_hierarchy(
        self, base_communities: List[Community], max_levels: int
    ) -> List[Community]:
        """
        Build hierarchical community structure.

        Groups similar communities into higher-level communities.
        """
        all_communities = list(base_communities)

        current_level = base_communities
        for level in range(1, max_levels):
            if len(current_level) <= 1:
                break

            # Simple hierarchical grouping: pair communities
            next_level = []
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Merge two communities
                    c1 = current_level[i]
                    c2 = current_level[i + 1]
                    parent = Community(
                        id=f"c_{level}_{len(next_level)}",
                        level=level,
                        entity_ids=c1.entity_ids + c2.entity_ids,
                        child_ids=[c1.id, c2.id],
                    )
                    c1.parent_id = parent.id
                    c2.parent_id = parent.id
                    next_level.append(parent)
                else:
                    # Odd one out, create single-child parent
                    c = current_level[i]
                    parent = Community(
                        id=f"c_{level}_{len(next_level)}",
                        level=level,
                        entity_ids=c.entity_ids,
                        child_ids=[c.id],
                    )
                    c.parent_id = parent.id
                    next_level.append(parent)

            all_communities.extend(next_level)
            current_level = next_level

        return all_communities


class CommunitySummarizer:
    """
    Generates summaries for communities.

    Uses LLM to create natural language descriptions of
    entity clusters for global search.

    Uses tier="fast" for summarization since this is a background task
    processing many communities.
    """

    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize summarizer.

        Args:
            llm_provider: LLM provider name (uses default from config if None)
        """
        self._llm_provider = llm_provider or "cohere"
        self._chat_client = None

    def _get_chat_client(self):
        """Lazy load chat client with 'fast' tier for background processing."""
        if self._chat_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._chat_client = get_llm_plugin(
                plugin_type="chat",
                plugin_name=self._llm_provider,
                tier="fast",  # Use fast model for background summarization
            )
        return self._chat_client

    def summarize_communities(
        self, communities: List[Community], graph: KnowledgeGraph
    ) -> List[Community]:
        """
        Generate summaries for all communities.

        Args:
            communities: List of communities to summarize
            graph: Knowledge graph with entity/relationship data

        Returns:
            Communities with summaries added
        """
        for community in communities:
            try:
                summary = self._summarize_community(community, graph)
                community.summary = summary
            except Exception as e:
                logger.warning(f"Failed to summarize community {community.id}: {e}")
                community.summary = self._fallback_summary(community, graph)

        return communities

    def _summarize_community(self, community: Community, graph: KnowledgeGraph) -> str:
        """Generate summary for a single community."""
        # Build entity descriptions
        entity_lines = []
        for entity_id in community.entity_ids[:20]:  # Limit for context
            entity = graph.get_entity(entity_id)
            if entity:
                line = f"- {entity.name} ({entity.type})"
                if entity.description:
                    line += f": {entity.description}"
                entity_lines.append(line)

        # Build relationship descriptions
        rel_lines = []
        entity_set = set(community.entity_ids)
        for rel in graph._relationships:
            if rel.source_id in entity_set and rel.target_id in entity_set:
                source = graph.get_entity(rel.source_id)
                target = graph.get_entity(rel.target_id)
                if source and target:
                    line = f"- {source.name} --[{rel.type}]--> {target.name}"
                    if rel.description:
                        line += f": {rel.description}"
                    rel_lines.append(line)
                    if len(rel_lines) >= 15:  # Limit relationships
                        break

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            entities="\n".join(entity_lines) if entity_lines else "No entities",
            relationships="\n".join(rel_lines) if rel_lines else "No relationships",
        )

        chat_client = self._get_chat_client()
        response = chat_client.chat([{"role": "user", "content": prompt}])

        return response.strip()

    def _fallback_summary(self, community: Community, graph: KnowledgeGraph) -> str:
        """Generate simple fallback summary without LLM."""
        entities = []
        for entity_id in community.entity_ids[:5]:
            entity = graph.get_entity(entity_id)
            if entity:
                entities.append(entity.name)

        if entities:
            return f"Community containing: {', '.join(entities)}"
        return f"Community with {len(community.entity_ids)} entities"
