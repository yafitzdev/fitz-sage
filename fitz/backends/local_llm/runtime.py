from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import urllib.request

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalLLMRuntimeConfig:
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    timeout: float = 1.0


class LocalLLMRuntime:
    """
    Local LLM runtime using Ollama.

    This runtime is USER-FACING.
    If unavailable, we terminate cleanly with instructions.
    """

    def __init__(self, cfg: LocalLLMRuntimeConfig) -> None:
        self._cfg = cfg
        self._client: Optional[Any] = None

    def _check_available(self) -> None:
        try:
            req = urllib.request.Request(self._cfg.base_url, method="GET")
            with urllib.request.urlopen(req, timeout=self._cfg.timeout):
                return
        except Exception as exc:
            logger.debug(
                "Local LLM runtime availability check failed",
                exc_info=exc,
            )
            raise SystemExit(
                "To enable local fallback:\n"
                "  1) Install Ollama: https://ollama.com\n"
                "  2) Run: ollama pull llama3\n"
                "  3) Ensure Ollama is running"
            )

    def llama(self) -> Any:
        if self._client is not None:
            return self._client

        self._check_available()

        try:
            import ollama  # type: ignore
        except Exception:
            raise SystemExit(
                "To enable local fallback:\n"
                "  1) Install Ollama: https://ollama.com\n"
                "  2) Run: ollama pull llama3\n"
                "  3) Ensure Ollama is running"
            )

        logger.info(f"{PIPELINE} Using local LLM model '{self._cfg.model}'")
        self._client = ollama
        return self._client
