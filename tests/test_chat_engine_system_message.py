from fitz_stack.llm.chat import ChatEngine

class Dummy:
    def chat(self, messages):
        return messages

def test_chat_engine_system_message():
    eng = ChatEngine(Dummy())
    msgs = [
        {"role": "system", "content": "system!"},
        {"role": "user", "content": "hello"},
    ]
    assert eng.chat(msgs) == msgs
