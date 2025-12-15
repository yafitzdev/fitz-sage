# fitz/core/llm/chat/plugins/local.py
from __future__ import annotations

from typing import Any

from fitz.backends.local_llm.chat import LocalChatLLM, LocalChatConfig
from fitz.core.llm.chat.base import ChatPlugin


class LocalChatClient(ChatPlugin):
    """
    Local fallback chat plugin.

    Thin adapter around fitz.backends.local_llm.chat.LocalChatLLM.
    """

    plugin_name = "local"
    plugin_type = "chat"
    availability = "local"

    def __init__(self, **kwargs: Any):
        cfg = LocalChatConfig(**kwargs)
        self._llm = LocalChatLLM(cfg)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        return self._llm.chat(messages)
