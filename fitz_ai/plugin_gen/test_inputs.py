# fitz_ai/plugin_gen/test_inputs.py
"""
Sample test inputs for functional validation of generated plugins.

Provides realistic test data for each plugin type to verify
that generated plugins actually work.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fitz_ai.plugin_gen.types import PluginType

# Sample markdown document for chunker testing
SAMPLE_MARKDOWN = """# Introduction

This is the introduction section of the document.
It contains some background information.

## Getting Started

To get started, follow these steps:

1. Install the package
2. Configure your settings
3. Run the main command

### Prerequisites

You need Python 3.10 or higher installed.

## Advanced Usage

This section covers advanced topics.

### Configuration Options

The following options are available:

- `debug`: Enable debug mode
- `verbose`: Show verbose output
- `timeout`: Set timeout in seconds

## Conclusion

This concludes the documentation.
"""

# Sample metadata for chunking
SAMPLE_META: Dict[str, Any] = {
    "source": "test_document.md",
    "title": "Test Document",
    "created_at": "2024-01-01",
}


# Mock chunk class for constraint testing
class MockChunk:
    """Mock chunk for testing constraints."""

    def __init__(
        self,
        content: str,
        source: str = "test.md",
        score: float = 0.9,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.source = source
        self.score = score
        self.metadata = metadata or {}


# Sample chunks for constraint testing
SAMPLE_CHUNKS = [
    MockChunk(
        content="The system uses a modular architecture with plugins.",
        source="architecture.md",
        score=0.95,
    ),
    MockChunk(
        content="Plugins are auto-discovered from the plugins directory.",
        source="plugins.md",
        score=0.88,
    ),
    MockChunk(
        content="Configuration is done through YAML files.",
        source="config.md",
        score=0.82,
    ),
]

# Sample query for constraint testing
SAMPLE_QUERY = "How does the plugin system work?"


def get_test_input(plugin_type: PluginType) -> Optional[Dict[str, Any]]:
    """
    Get test input for a given plugin type.

    Args:
        plugin_type: The type of plugin to get test input for

    Returns:
        Dictionary with test inputs, or None if no test input available
    """
    inputs = {
        PluginType.CHUNKER: {
            "text": SAMPLE_MARKDOWN,
            "meta": SAMPLE_META,
        },
        PluginType.CONSTRAINT: {
            "query": SAMPLE_QUERY,
            "chunks": SAMPLE_CHUNKS,
        },
        # Reader requires actual files, skip for now
        PluginType.READER: None,
    }

    return inputs.get(plugin_type)


def get_expected_behavior(plugin_type: PluginType) -> str:
    """
    Get description of expected behavior for a plugin type.

    Used in LLM prompts to explain what the generated plugin should do.
    """
    behaviors = {
        PluginType.CHUNKER: (
            "The chunk_text method should:\n"
            "- Split the input text into semantic chunks\n"
            "- Preserve document structure where possible\n"
            "- Return a list of Chunk objects with content and metadata\n"
            "- Handle edge cases (empty text, very long text)"
        ),
        PluginType.READER: (
            "The ingest method should:\n"
            "- Accept a file path and yield RawDocument objects\n"
            "- Handle the specific file format appropriately\n"
            "- Extract text content and relevant metadata\n"
            "- Support both single files and directories"
        ),
        PluginType.CONSTRAINT: (
            "The apply method should:\n"
            "- Evaluate if retrieved chunks support answering the query\n"
            "- Return ConstraintResult with allow_decisive_answer boolean\n"
            "- Optionally include a reason explaining the decision\n"
            "- Be conservative - prefer 'I don't know' over hallucination"
        ),
        PluginType.LLM_CHAT: (
            "The plugin should:\n"
            "- Define the API endpoint and authentication\n"
            "- Specify request/response format mappings\n"
            "- Handle streaming if supported\n"
            "- Include proper error handling configuration"
        ),
        PluginType.LLM_EMBEDDING: (
            "The plugin should:\n"
            "- Define the embedding API endpoint\n"
            "- Specify the embedding dimension\n"
            "- Handle batch requests efficiently\n"
            "- Include proper authentication"
        ),
        PluginType.LLM_RERANK: (
            "The plugin should:\n"
            "- Define the reranking API endpoint\n"
            "- Specify request format for query and documents\n"
            "- Parse response to extract relevance scores\n"
            "- Handle proper authentication"
        ),
        PluginType.VECTOR_DB: (
            "The plugin should:\n"
            "- Define connection parameters (host, port, etc.)\n"
            "- Specify collection/index operations\n"
            "- Define search and upsert operations\n"
            "- Include proper authentication if needed"
        ),
        PluginType.RETRIEVAL: (
            "The plugin should:\n"
            "- Define retrieval strategy configuration\n"
            "- Specify which retrieval components to use\n"
            "- Configure any strategy-specific parameters"
        ),
    }

    return behaviors.get(plugin_type, "Follow the standard plugin protocol.")


__all__ = [
    "get_test_input",
    "get_expected_behavior",
    "SAMPLE_MARKDOWN",
    "SAMPLE_META",
    "SAMPLE_CHUNKS",
    "SAMPLE_QUERY",
    "MockChunk",
]
