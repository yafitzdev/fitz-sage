# fitz/engines/clara/config/schema.py
"""
CLaRa Engine Configuration Schema.

This module defines the configuration structure for the CLaRa engine,
which uses continuous latent reasoning for retrieval-augmented generation.
"""

from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class ClaraModelConfig:
    """
    Configuration for CLaRa model loading.

    CLaRa has three model variants:
    - Base: Compression pretraining (CLaRa-7B-Base)
    - Instruct: Instruction tuning (CLaRa-7B-Instruct)
    - E2E: End-to-end retrieval + generation (CLaRa-7B-E2E)
    """

    model_name_or_path: str = "apple/CLaRa-7B-E2E"
    """HuggingFace model path or local path."""

    variant: Literal["base", "instruct", "e2e"] = "e2e"
    """Which CLaRa variant to use. E2E recommended for full RAG."""

    device: str = "cuda"
    """Device to load model on ('cuda', 'cpu', 'mps')."""

    torch_dtype: str = "bfloat16"
    """Torch dtype for model weights ('float32', 'float16', 'bfloat16')."""

    trust_remote_code: bool = True
    """Whether to trust remote code from HuggingFace."""

    load_in_8bit: bool = False
    """Whether to use 8-bit quantization for memory efficiency."""

    load_in_4bit: bool = False
    """Whether to use 4-bit quantization for memory efficiency."""


@dataclass
class ClaraCompressionConfig:
    """
    Configuration for document compression.

    CLaRa compresses documents into continuous memory tokens,
    achieving 16x-128x compression while preserving semantics.
    """

    compression_rate: int = 16
    """Compression rate (4, 16, 32, 64, 128). Higher = more compression."""

    doc_max_length: int = 256
    """Maximum document length before compression."""

    num_memory_tokens: Optional[int] = None
    """Number of memory tokens per document. Auto-calculated if None."""


@dataclass
class ClaraRetrievalConfig:
    """
    Configuration for latent space retrieval.

    CLaRa performs retrieval by computing cosine similarity between
    query embeddings and compressed document embeddings in latent space.
    """

    top_k: int = 5
    """Number of documents to retrieve for generation."""

    candidate_pool_size: int = 20
    """Number of candidate documents to consider for reranking."""

    use_differentiable_topk: bool = True
    """Whether to use differentiable top-k (for training). Disable for inference."""


@dataclass
class ClaraGenerationConfig:
    """
    Configuration for answer generation.
    """

    max_new_tokens: int = 256
    """Maximum tokens to generate in answer."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Nucleus sampling probability."""

    do_sample: bool = True
    """Whether to use sampling (vs greedy decoding)."""


@dataclass
class ClaraConfig:
    """
    Complete configuration for the CLaRa engine.

    This is the main config object that ClaraEngine expects.

    Examples:
        Default config:
        >>> config = ClaraConfig()

        Custom config:
        >>> config = ClaraConfig(
        ...     model=ClaraModelConfig(model_name_or_path="apple/CLaRa-7B-Instruct"),
        ...     compression=ClaraCompressionConfig(compression_rate=32),
        ...     retrieval=ClaraRetrievalConfig(top_k=10),
        ... )
    """

    model: ClaraModelConfig = field(default_factory=ClaraModelConfig)
    """Model loading configuration."""

    compression: ClaraCompressionConfig = field(default_factory=ClaraCompressionConfig)
    """Document compression configuration."""

    retrieval: ClaraRetrievalConfig = field(default_factory=ClaraRetrievalConfig)
    """Retrieval configuration."""

    generation: ClaraGenerationConfig = field(default_factory=ClaraGenerationConfig)
    """Generation configuration."""

    # Knowledge base settings
    knowledge_base_path: Optional[str] = None
    """Path to pre-compressed knowledge base (optional)."""

    cache_compressed_docs: bool = True
    """Whether to cache compressed document embeddings."""


def load_clara_config(config_path: Optional[str] = None) -> ClaraConfig:
    """
    Load CLaRa configuration from file or return defaults.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        ClaraConfig object
    """
    if config_path is None:
        return ClaraConfig()

    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse nested configs
    clara_data = data.get("clara", data)

    return ClaraConfig(
        model=ClaraModelConfig(**clara_data.get("model", {})),
        compression=ClaraCompressionConfig(**clara_data.get("compression", {})),
        retrieval=ClaraRetrievalConfig(**clara_data.get("retrieval", {})),
        generation=ClaraGenerationConfig(**clara_data.get("generation", {})),
        knowledge_base_path=clara_data.get("knowledge_base_path"),
        cache_compressed_docs=clara_data.get("cache_compressed_docs", True),
    )