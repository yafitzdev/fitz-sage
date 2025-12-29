# fitz_ai/engines/graphrag/search/global_search.py
"""
Global Search for GraphRAG.

Community-based search that uses hierarchical summaries to
answer broad questions about the entire knowledge base.
"""

from dataclasses import dataclass
from typing import List, Optional

from fitz_ai.engines.graphrag.config.schema import GraphSearchConfig
from fitz_ai.engines.graphrag.graph.storage import Community, KnowledgeGraph
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GlobalSearchResult:
    """Result from global search."""

    communities: List[Community]
    context: str
    scores: List[float]


class GlobalSearch:
    """
    Global search over the knowledge graph.

    Uses community summaries to answer questions that require
    understanding of the entire knowledge base.
    """

    def __init__(
        self,
        config: GraphSearchConfig,
        embedding_provider: Optional[str] = None,
    ):
        """
        Initialize global search.

        Args:
            config: Search configuration
            embedding_provider: Embedding provider name
        """
        self.config = config
        self._embedding_provider = embedding_provider or "cohere"
        self._embed_client = None
        self._community_embeddings: dict = {}

    def _get_embed_client(self):
        """Lazy load embedding client."""
        if self._embed_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._embed_client = get_llm_plugin(
                plugin_type="embedding",
                plugin_name=self._embedding_provider,
            )
        return self._embed_client

    def index_communities(self, graph: KnowledgeGraph) -> None:
        """
        Build embedding index for community summaries.

        Args:
            graph: Knowledge graph with communities
        """
        embed_client = self._get_embed_client()
        self._community_embeddings.clear()

        communities = graph.get_all_communities()
        if not communities:
            return

        # Filter to communities with summaries
        communities_with_summaries = [c for c in communities if c.summary]
        if not communities_with_summaries:
            return

        # Embed summaries using embed_batch
        summaries = [c.summary for c in communities_with_summaries]
        all_embeddings = embed_client.embed_batch(summaries)

        for community, embedding in zip(communities_with_summaries, all_embeddings):
            self._community_embeddings[community.id] = embedding

        logger.info(f"Indexed {len(communities_with_summaries)} community summaries")

    def search(
        self,
        query: str,
        graph: KnowledgeGraph,
        top_k: Optional[int] = None,
    ) -> GlobalSearchResult:
        """
        Search for relevant communities.

        Args:
            query: Search query
            graph: Knowledge graph to search
            top_k: Number of communities to retrieve

        Returns:
            GlobalSearchResult with communities and context
        """
        top_k = top_k or self.config.global_top_k

        if not self._community_embeddings:
            self.index_communities(graph)

        if not self._community_embeddings:
            # No communities indexed
            return GlobalSearchResult(communities=[], context="", scores=[])

        # Embed query
        embed_client = self._get_embed_client()
        query_embedding = embed_client.embed(query)

        # Compute similarities
        similarities = []
        for community_id, comm_emb in self._community_embeddings.items():
            sim = self._cosine_similarity(query_embedding, comm_emb)
            similarities.append((community_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top communities
        top_communities = []
        top_scores = []
        for community_id, score in similarities[:top_k]:
            community = graph.get_community(community_id)
            if community:
                top_communities.append(community)
                top_scores.append(score)

        # Build context from community summaries
        context = self._build_context(top_communities, graph)

        return GlobalSearchResult(
            communities=top_communities,
            context=context,
            scores=top_scores,
        )

    def _build_context(self, communities: List[Community], graph: KnowledgeGraph) -> str:
        """Build context string from communities."""
        lines = ["## Relevant Knowledge Summaries\n"]

        for i, community in enumerate(communities, 1):
            lines.append(f"### Summary {i}")
            lines.append(community.summary)
            lines.append("")

            # Optionally include key entities
            key_entities = []
            for entity_id in community.entity_ids[:5]:
                entity = graph.get_entity(entity_id)
                if entity:
                    key_entities.append(f"{entity.name} ({entity.type})")

            if key_entities:
                lines.append(f"Key entities: {', '.join(key_entities)}")
                lines.append("")

        return "\n".join(lines)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class HybridSearch:
    """
    Combines local and global search.

    Uses both entity-level and community-level retrieval
    for comprehensive context.
    """

    def __init__(
        self,
        config: GraphSearchConfig,
        embedding_provider: Optional[str] = None,
    ):
        """
        Initialize hybrid search.

        Args:
            config: Search configuration
            embedding_provider: Embedding provider name
        """
        from fitz_ai.engines.graphrag.search.local import LocalSearch

        self.config = config
        self.local_search = LocalSearch(config, embedding_provider)
        self.global_search = GlobalSearch(config, embedding_provider)

    def index(self, graph: KnowledgeGraph) -> None:
        """Index both entities and communities."""
        self.local_search.index_entities(graph)
        self.global_search.index_communities(graph)

    def search(
        self,
        query: str,
        graph: KnowledgeGraph,
        local_weight: float = 0.5,
    ) -> str:
        """
        Search using both local and global methods.

        Args:
            query: Search query
            graph: Knowledge graph
            local_weight: Weight for local vs global (0-1)

        Returns:
            Combined context string
        """
        # Get local results
        local_result = self.local_search.search(query, graph, top_k=self.config.local_top_k)

        # Get global results
        global_result = self.global_search.search(query, graph, top_k=self.config.global_top_k)

        # Combine contexts
        context_parts = []

        if global_result.context:
            context_parts.append(global_result.context)

        if local_result.context:
            context_parts.append("\n## Specific Entity Information\n")
            context_parts.append(local_result.context)

        return "\n".join(context_parts)
