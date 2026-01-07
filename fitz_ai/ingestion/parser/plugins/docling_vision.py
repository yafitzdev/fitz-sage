# fitz_ai/ingestion/parser/plugins/docling_vision.py
"""
Docling parser with VLM-powered figure description.

This parser extends DoclingParser to automatically use VLM (Vision Language Model)
for describing figures and images in documents. It reads the vision provider
configuration from the engine config's `vision:` section.

Use this parser when you want AI-generated descriptions of charts, graphs,
diagrams, and other visual content in PDFs and documents.

Usage:
    The parser is selected via the chunking config:

        chunking:
          default:
            parser: docling_vision  # Uses VLM for figure description

    Requires vision provider to be configured:

        vision:
          plugin_name: cohere  # or openai, anthropic, local_ollama
          kwargs: {}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set

from fitz_ai.ingestion.parser.plugins.docling import DOCLING_EXTENSIONS, DoclingParser


@dataclass
class DoclingVisionParser(DoclingParser):
    """
    Docling parser with automatic VLM integration for figure description.

    This parser automatically loads the vision client from the engine config's
    `vision:` section. Figures and images detected by Docling are sent to the
    configured VLM for description, replacing "[Figure]" placeholders with
    actual content descriptions.

    The vision client is loaded lazily on first use to avoid initialization
    overhead when not needed.
    """

    plugin_name: str = field(default="docling_vision", repr=False)
    supported_extensions: Set[str] = field(default_factory=lambda: DOCLING_EXTENSIONS)

    # Vision client loaded lazily from config
    _vision_client_loaded: bool = field(default=False, repr=False)

    def _ensure_vision_client(self) -> None:
        """Load vision client from config if not already loaded."""
        if self._vision_client_loaded:
            return

        self._vision_client_loaded = True

        try:
            from fitz_ai.cli.context import CLIContext

            ctx = CLIContext.load()
            config = ctx.raw_config
            vision_cfg = config.get("vision", {})

            if vision_cfg.get("plugin_name"):
                from fitz_ai.llm.registry import get_llm_plugin

                vision_kwargs = vision_cfg.get("kwargs", {})
                self.vision_client = get_llm_plugin(
                    plugin_type="vision",
                    plugin_name=vision_cfg["plugin_name"],
                    tier="fast",
                    **vision_kwargs,
                )
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to load vision client for docling_vision: {e}")

    def _describe_image_with_vlm(self, item) -> str | None:
        """Override to ensure vision client is loaded before use."""
        self._ensure_vision_client()
        return super()._describe_image_with_vlm(item)


__all__ = ["DoclingVisionParser", "DOCLING_EXTENSIONS"]
