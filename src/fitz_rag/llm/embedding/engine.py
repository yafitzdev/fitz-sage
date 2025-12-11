from __future__ import annotations

from fitz_rag.exceptions.retriever import EmbeddingError
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import EMBEDDING

logger = get_logger(__name__)


class EmbeddingEngine:
    """
    Wraps a text embedding plugin and handles:
    - logging
    - error translation
    - future multi-embedding fusion
    """

    def __init__(self, plugin):
        self.plugin = plugin

    def embed(self, text: str):
        logger.debug(f"{EMBEDDING} Embedding text ({len(text)} chars)")

        try:
            return self.plugin.embed(text)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding plugin failed: {e}")
            raise EmbeddingError("Embedding request failed") from e
