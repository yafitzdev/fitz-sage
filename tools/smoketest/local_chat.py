from fitz.core.llm.chat.plugins.local import LocalChatClient
from fitz.core.exceptions.llm import LLMError


def main() -> None:
    llm = LocalChatClient()
    llm.chat([
        {"role": "user", "content": "hello"}
    ])


if __name__ == "__main__":
    try:
        main()
    except LLMError as e:
        print(e)
