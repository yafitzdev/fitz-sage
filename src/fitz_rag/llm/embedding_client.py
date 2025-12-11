from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional
import os

from fitz_rag.exceptions.retriever import EmbeddingError

try:  # Cohere is optional at import time
    import cohere
except ImportError:  # pragma: no cover - handled lazily at runtime
    cohere = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# EmbeddingClient protocol
# ---------------------------------------------------------------------------


class EmbeddingClient(Protocol):
    """
    Minimal interface for embedding providers.

    Any concrete implementation must implement `embed(text) -> List[float]`.
    """

    def embed(self, text: str) -> List[float]:
        ...


# ---------------------------------------------------------------------------
# Cohere implementation
# ---------------------------------------------------------------------------


@dataclass
class CohereEmbeddingClient:
    """
    Cohere embedding wrapper.

    Defaults:
      - model: embed-english-v3.0
      - input_type: search_query
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None  # None = model default

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError(
                "cohere is not installed. Run `pip install cohere` inside your environment."
            )

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it in your environment or pass api_key=... to CohereEmbeddingClient."
            )

        self.model = (
            self.model
            or os.getenv("COHERE_EMBED_MODEL")
            or "embed-english-v3.0"
        )
        self.input_type = (
            self.input_type
            or os.getenv("COHERE_EMBED_INPUT_TYPE")
            or "search_query"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise EmbeddingError("Failed to initialize Cohere embedding client") from e

    def embed(self, text: str) -> List[float]:
        """Create a single embedding with structured exception handling."""
        kwargs = {
            "texts": [text],
            "model": self.model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension

        try:
            res = self._client.embed(**kwargs)
            return res.embeddings.float[0]
        except Exception as e:
            raise EmbeddingError(f"Failed to embed text: {text!r}") from e


# ---------------------------------------------------------------------------
# Dummy implementation (for tests)
# ---------------------------------------------------------------------------


@dataclass
class DummyEmbeddingClient:
    dim: int = 10

    def embed(self, text: str) -> List[float]:
        base = abs(hash(text)) % 997
        return [(base + i) / 997.0 for i in range(self.dim)]
