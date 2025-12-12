# tests/test_chat_engine_registry.py

from fitz_stack.llm.chat import ChatEngine
from fitz_stack.llm.chat.base import ChatPlugin

# Create a temporary plugin
class DummyChat(ChatPlugin):
    plugin_name = "dummy_chat"
    def chat(self, messages): return "ok"

def test_chat_engine_registry_loads_plugin(monkeypatch):
    import fitz_stack.llm.chat.registry as reg
    reg.CHAT_REGISTRY["dummy_chat"] = DummyChat

    engine = ChatEngine.from_name("dummy_chat")
    assert engine.chat_text("hi") == "ok"
