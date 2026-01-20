# fitz_ai/ingestion/enrichment/prompts/__init__.py
"""
Externalized prompts for enrichment modules.

Prompts are stored as .txt files and loaded via helper functions.
"""

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_chunk_prompt(name: str) -> str:
    """
    Load a chunk-level prompt by name.

    Args:
        name: Prompt name (e.g., "summary", "keywords", "entities")

    Returns:
        Prompt text content
    """
    prompt_path = _PROMPTS_DIR / "chunk" / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


__all__ = ["load_chunk_prompt"]
