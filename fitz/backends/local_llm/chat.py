# fitz/backends/local_llm/chat.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CHAT

from fitz.backends.local_llm.runtime import (
    LocalLLMRuntime,
    LocalLLMRuntimeConfig,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class LocalChatConfig:
    """
    Baseline local chat settings.

    Keep this intentionally minimal: this is not a “model zoo”.
    """

    max_tokens: int = 256
    temperature: float = 0.2


class LocalChatLLM:
    """
    Local ChatLLM adapter.

    Expected message format:
      - Sequence of dict-like objects with keys: 'role', 'content'
      - Roles: 'system'|'user'|'assistant' (best-effort)
    """

    def __init__(
        self,
        cfg: LocalChatConfig | None = None,
        runtime_cfg: LocalLLMRuntimeConfig | None = None,
    ) -> None:
        self._cfg = cfg or LocalChatConfig()
        self._rt = LocalLLMRuntime(runtime_cfg or LocalLLMRuntimeConfig())

    def chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        logger.info(f"{CHAT} Using local chat model (baseline quality)")

        llm = self._rt.llama()

        # Ollama-style chat
        if hasattr(llm, "chat"):
            resp: Any = llm.chat(
                model=self._rt._cfg.model,
                messages=[
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages
                ],
                options={
                    "temperature": self._cfg.temperature,
                    "num_predict": self._cfg.max_tokens,
                },
            )
            return _extract_chat_text(resp)

        raise RuntimeError("Unsupported local LLM client")


def _extract_chat_text(resp: Any) -> str:
    try:
        message = resp.get("message") or {}
        content = message.get("content")
        if content is not None:
            return str(content)
    except Exception:
        pass
    return str(resp)
