# fitz_ai/engines/graphrag/config/schema.py
"""
GraphRAG Configuration Schema.

Configuration dataclasses for the GraphRAG engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class GraphExtractionConfig:
    """Configuration for entity and relationship extraction."""

    # Maximum entities to extract per chunk
    max_entities_per_chunk: int = 20

    # Maximum relationships to extract per chunk
    max_relationships_per_chunk: int = 30

    # Entity types to extract (empty = all types)
    entity_types: List[str] = field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "event",
            "concept",
            "technology",
            "product",
            "date",
        ]
    )

    # Relationship types to extract (empty = all types)
    relationship_types: List[str] = field(default_factory=list)

    # Whether to include entity descriptions
    include_descriptions: bool = True


@dataclass
class GraphCommunityConfig:
    """Configuration for community detection."""

    # Algorithm: "louvain" or "leiden"
    algorithm: Literal["louvain", "leiden"] = "louvain"

    # Resolution parameter (higher = more communities)
    resolution: float = 1.0

    # Minimum community size
    min_community_size: int = 2

    # Maximum hierarchy levels for community summaries
    max_hierarchy_levels: int = 2


@dataclass
class GraphSearchConfig:
    """Configuration for graph search."""

    # Default search mode
    default_mode: Literal["local", "global", "hybrid"] = "local"

    # Number of entities to retrieve in local search
    local_top_k: int = 10

    # Number of communities to retrieve in global search
    global_top_k: int = 5

    # Whether to include entity relationships in context
    include_relationships: bool = True

    # Maximum context tokens for generation
    max_context_tokens: int = 4000


@dataclass
class GraphStorageConfig:
    """Configuration for graph storage."""

    # Storage backend: "memory" or "file"
    backend: Literal["memory", "file"] = "memory"

    # Path for file-based storage
    storage_path: Optional[str] = None


@dataclass
class GraphRAGConfig:
    """
    Main configuration for GraphRAG engine.

    Examples:
        Default config:
        >>> config = GraphRAGConfig()

        Custom config:
        >>> config = GraphRAGConfig(
        ...     extraction=GraphExtractionConfig(max_entities_per_chunk=30),
        ...     community=GraphCommunityConfig(algorithm="leiden"),
        ...     search=GraphSearchConfig(default_mode="global"),
        ... )
    """

    extraction: GraphExtractionConfig = field(default_factory=GraphExtractionConfig)
    community: GraphCommunityConfig = field(default_factory=GraphCommunityConfig)
    search: GraphSearchConfig = field(default_factory=GraphSearchConfig)
    storage: GraphStorageConfig = field(default_factory=GraphStorageConfig)

    # LLM provider for extraction and summarization (uses fitz default if not set)
    llm_provider: Optional[str] = None

    # Embedding provider for entity similarity (uses fitz default if not set)
    embedding_provider: Optional[str] = None


def get_default_config_path() -> Path:
    """Get path to the package default GraphRAG config file."""
    return Path(__file__).parent / "default.yaml"


def get_user_config_path() -> Path:
    """Get path to user's GraphRAG config file (.fitz/config/graphrag.yaml)."""
    from fitz_ai.core.paths import FitzPaths

    return FitzPaths.engine_config("graphrag")


def load_graphrag_config(config_path: Optional[str] = None) -> GraphRAGConfig:
    """
    Load GraphRAG configuration from a YAML file.

    Resolution order:
    1. Explicit config_path if provided
    2. User config at .fitz/config/graphrag.yaml
    3. Package default at fitz_ai/engines/graphrag/config/default.yaml

    Args:
        config_path: Path to YAML config file. If None, uses resolution order.

    Returns:
        GraphRAGConfig object
    """
    if config_path is None:
        # Check for user config first
        user_config = get_user_config_path()
        if user_config.exists():
            config_path = str(user_config)
        else:
            # Fall back to package default
            config_path = str(get_default_config_path())

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    # Extract graphrag section if present
    graphrag_data = data.get("graphrag", data)

    return GraphRAGConfig(
        extraction=GraphExtractionConfig(**graphrag_data.get("extraction", {})),
        community=GraphCommunityConfig(**graphrag_data.get("community", {})),
        search=GraphSearchConfig(**graphrag_data.get("search", {})),
        storage=GraphStorageConfig(**graphrag_data.get("storage", {})),
        llm_provider=graphrag_data.get("llm_provider"),
        embedding_provider=graphrag_data.get("embedding_provider"),
    )
