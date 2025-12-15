from fitz.core.llm.embedding.plugins.local import LocalEmbeddingClient
from fitz.core.exceptions.llm import LLMError


def main() -> None:
    embedder = LocalEmbeddingClient()
    embedder.embed("hello world")


if __name__ == "__main__":
    try:
        main()
    except LLMError as e:
        print(e)
