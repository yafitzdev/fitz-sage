from fitz.core.llm.rerank.plugins.local import LocalRerankClient
from fitz.core.models.chunk import Chunk
from fitz.core.exceptions.llm import LLMError


def main() -> None:
    reranker = LocalRerankClient()
    reranker.rerank(
        query="test",
        chunks=[
            Chunk(
                id="1",
                doc_id="d",
                chunk_index=0,
                content="hello",
                metadata={},
            )
        ],
    )


if __name__ == "__main__":
    try:
        main()
    except LLMError as e:
        print(e)
