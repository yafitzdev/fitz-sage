# fitz_ai/engines/graphrag/search/local.py
"""
Local Search for GraphRAG.

Entity-centric search that retrieves relevant entities and their
local neighborhood for answering specific questions.
"""

from dataclasses import dataclass
from typing import List, Optional

from fitz_ai.engines.graphrag.config.schema import GraphSearchConfig
from fitz_ai.engines.graphrag.graph.storage import Entity, KnowledgeGraph
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LocalSearchResult:
    """Result from local search."""

    entities: List[Entity]
    context: str
    scores: List[float]


class LocalSearch:
    """
    Local search over the knowledge graph.

    Finds relevant entities based on query and expands to
    their local neighborhood for context.
    """

    def __init__(
        self,
        config: GraphSearchConfig,
        embedding_provider: Optional[str] = None,
    ):
        """
        Initialize local search.

        Args:
            config: Search configuration
            embedding_provider: Embedding provider name
        """
        self.config = config
        self._embedding_provider = embedding_provider or "cohere"
        self._embed_client = None
        self._entity_embeddings: dict = {}

    def _get_embed_client(self):
        """Lazy load embedding client."""
        if self._embed_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._embed_client = get_llm_plugin(
                plugin_type="embedding",
                plugin_name=self._embedding_provider,
            )
        return self._embed_client

    def index_entities(self, graph: KnowledgeGraph) -> None:
        """
        Build embedding index for entities.

        Args:
            graph: Knowledge graph to index
        """
        embed_client = self._get_embed_client()
        self._entity_embeddings.clear()

        # Batch embed all entities
        entities = list(graph._entities.values())
        if not entities:
            return

        # Create text representations for embedding
        texts = []
        for entity in entities:
            text = f"{entity.name} ({entity.type})"
            if entity.description:
                text += f": {entity.description}"
            texts.append(text)

        # Embed in batches using embed_batch
        all_embeddings = embed_client.embed_batch(texts)

        # Store embeddings
        for entity, embedding in zip(entities, all_embeddings):
            self._entity_embeddings[entity.id] = embedding

        logger.info(f"Indexed {len(entities)} entities for local search")

    def search(
        self,
        query: str,
        graph: KnowledgeGraph,
        top_k: Optional[int] = None,
    ) -> LocalSearchResult:
        """
        Search for relevant entities.

        Args:
            query: Search query
            graph: Knowledge graph to search
            top_k: Number of entities to retrieve

        Returns:
            LocalSearchResult with entities and context
        """
        top_k = top_k or self.config.local_top_k

        if not self._entity_embeddings:
            self.index_entities(graph)

        if not self._entity_embeddings:
            # No entities to search
            return LocalSearchResult(entities=[], context="", scores=[])

        # Embed query
        embed_client = self._get_embed_client()
        query_embedding = embed_client.embed(query)

        # Compute similarities
        similarities = []
        for entity_id, entity_emb in self._entity_embeddings.items():
            sim = self._cosine_similarity(query_embedding, entity_emb)
            similarities.append((entity_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top entities
        top_entities = []
        top_scores = []
        for entity_id, score in similarities[:top_k]:
            entity = graph.get_entity(entity_id)
            if entity:
                top_entities.append(entity)
                top_scores.append(score)

        # Expand to neighborhood
        if self.config.include_relationships:
            expanded_ids = set(e.id for e in top_entities)
            for entity in top_entities[:5]:  # Expand top 5
                neighbors = graph.get_neighbors(entity.id, max_hops=1)
                expanded_ids.update(neighbors)

            # Build context from expanded set
            context = graph.to_context_string(
                entity_ids=list(expanded_ids),
                include_relationships=True,
            )
        else:
            context = graph.to_context_string(
                entity_ids=[e.id for e in top_entities],
                include_relationships=False,
            )

        return LocalSearchResult(
            entities=top_entities,
            context=context,
            scores=top_scores,
        )

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
