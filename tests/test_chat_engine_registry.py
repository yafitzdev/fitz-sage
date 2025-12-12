from fitz_stack.llm.registry import register_llm_plugin
from fitz_stack.llm.chat import ChatEngine
from fitz_stack.llm.chat.base import ChatPlugin

class DummyChat(ChatPlugin):
    plugin_name = "dummy_chat"

    def chat(self, messages):
        return "ok"

def test_chat_engine_registry_loads_plugin():
    register_llm_plugin(
        DummyChat,
        plugin_name="dummy_chat",
        plugin_type="chat",
    )

    engine = ChatEngine.from_name("dummy_chat")
    assert engine.chat([{"role": "user", "content": "hello"}]) == "ok"
