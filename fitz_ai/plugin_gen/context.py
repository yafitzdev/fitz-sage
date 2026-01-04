# fitz_ai/plugin_gen/context.py
"""
Context builder for LLM plugin generation.

Loads existing plugin examples and schema definitions to build
prompts that help the LLM generate correct, working plugins.
"""

from __future__ import annotations

import logging
from typing import Optional

from fitz_ai.plugin_gen.test_inputs import get_expected_behavior
from fitz_ai.plugin_gen.types import PluginType

logger = logging.getLogger(__name__)


# =============================================================================
# Example Loading
# =============================================================================


def _load_yaml_example(plugin_type: PluginType) -> Optional[str]:
    """Load an example YAML plugin for reference."""
    examples_dir = plugin_type.get_example_plugins_dir()

    if not examples_dir.exists():
        logger.debug(f"Examples directory does not exist: {examples_dir}")
        return None

    # Find first .yaml file
    yaml_files = list(examples_dir.glob("*.yaml"))
    if not yaml_files:
        logger.debug(f"No YAML files found in {examples_dir}")
        return None

    # Prefer openai.yaml or first file
    for preferred in ["openai.yaml", "qdrant.yaml", "dense.yaml"]:
        for f in yaml_files:
            if f.name == preferred:
                return f.read_text(encoding="utf-8")

    return yaml_files[0].read_text(encoding="utf-8")


def _load_python_example(plugin_type: PluginType) -> Optional[str]:
    """Load an example Python plugin for reference."""
    examples_dir = plugin_type.get_example_plugins_dir()

    if not examples_dir.exists():
        logger.debug(f"Examples directory does not exist: {examples_dir}")
        return None

    # Find first .py file that's not __init__
    py_files = [f for f in examples_dir.glob("*.py") if not f.name.startswith("_")]
    if not py_files:
        logger.debug(f"No Python files found in {examples_dir}")
        return None

    # Prefer simple.py or conflict_aware.py
    for preferred in ["simple.py", "conflict_aware.py", "local_fs.py"]:
        for f in py_files:
            if f.name == preferred:
                return f.read_text(encoding="utf-8")

    return py_files[0].read_text(encoding="utf-8")


def load_example_plugin(plugin_type: PluginType) -> Optional[str]:
    """Load an example plugin for the given type."""
    if plugin_type.is_yaml:
        return _load_yaml_example(plugin_type)
    else:
        return _load_python_example(plugin_type)


# =============================================================================
# Schema Information
# =============================================================================


def get_yaml_schema_info(plugin_type: PluginType) -> str:
    """Get schema requirements for YAML plugins."""
    schemas = {
        PluginType.LLM_CHAT: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "chat"
- version: string (e.g., "1.0")
- provider:
    - name: string (provider identifier)
    - base_url: string (API base URL, must start with http:// or https://)
- auth:
    - type: "bearer" | "header" | "none"
    - env_vars: list of environment variable names for API key
    - header_name: string (default "Authorization")
    - header_format: string with {key} placeholder
- endpoint:
    - path: string (must start with /)
    - method: "POST" | "GET"
    - timeout: integer (seconds)
- defaults:
    - models:
        - smart: string (model for high-quality responses)
        - fast: string (model for speed)
        - balanced: string (cost-effective model)
    - temperature: float (0.0 - 1.0)
- request:
    - messages_transform: "openai_chat" | "anthropic_chat" | "cohere_chat" | "gemini_chat" | "ollama_chat"
    - param_map: dict mapping param names
- response:
    - content_path: JSONPath to extract response text (e.g., "choices[0].message.content")
    - metadata_paths: optional dict for token counts etc.
""",
        PluginType.LLM_EMBEDDING: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "embedding"
- version: string
- provider: same as chat
- auth: same as chat (MUST have env_vars with API key variable name)
- endpoint: same as chat
- defaults:
    - models:
        - smart: string (embedding model name)
        - fast: string
        - balanced: string
- request:
    - input_field: string (field name for input text)
    - input_wrap: "list" | "string" | "object"
- response:
    - embeddings_path: JSONPath to embedding vector (e.g., "data[0].embedding")
""",
        PluginType.LLM_RERANK: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "rerank"
- version: string
- provider: same as chat
- auth: same as chat (MUST have env_vars with API key variable name)
- endpoint: same as chat
- defaults:
    - models:
        - smart: string (rerank model name)
- request:
    - query_field: string (field name for query)
    - documents_field: string (field name for documents list)
- response:
    - results_path: JSONPath to results array
    - result_index_path: path to index within result
    - result_score_path: path to relevance score
""",
        PluginType.VECTOR_DB: """
Required fields:
- name: string (unique identifier)
- type: string (e.g., "qdrant", "pinecone", "milvus")
- connection:
    - host: string (can be {ENV_VAR} placeholder)
    - port: integer
- operations:
    - create_collection: HTTP operation spec
    - delete_collection: HTTP operation spec
    - upsert: HTTP operation spec
    - search: HTTP operation spec
""",
        PluginType.RETRIEVAL: """
Required fields:
- name: string (strategy name)
- type: string (e.g., "dense", "hybrid")
- Any strategy-specific configuration
""",
    }

    return schemas.get(plugin_type, "Follow standard YAML plugin format.")


def get_python_protocol_info(plugin_type: PluginType) -> str:
    """Get protocol requirements for Python plugins."""
    protocols = {
        PluginType.CHUNKER: """
Required class structure:
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
from fitz_ai.core.chunk import Chunk

@dataclass
class MyChunker:
    plugin_name: str = field(default="my_chunker", repr=False)
    # Add configuration parameters here

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        '''
        Split text into chunks.

        Args:
            text: The text content to chunk
            base_meta: Metadata dict with doc_id, source_file, etc.

        Returns:
            List of Chunk objects
        '''
        # Implementation here
        pass
```

The Chunk class has these attributes:
- id: str (unique chunk ID, typically "{doc_id}:{chunk_index}")
- doc_id: str (document identifier)
- chunk_index: int (position in document)
- content: str (chunk text)
- metadata: Dict[str, Any] (inherited from base_meta)
""",
        PluginType.READER: """
Required class structure:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

@dataclass
class MyReader:
    plugin_name: str = field(default="my_reader", repr=False)

    def ingest(self, path: Path) -> Iterable[RawDocument]:
        '''
        Read files and yield RawDocument objects.

        Args:
            path: Path to file or directory

        Yields:
            RawDocument objects with content and metadata
        '''
        pass
```

RawDocument has:
- content: str (file text content)
- metadata: Dict[str, Any] (source info)
""",
        PluginType.CONSTRAINT: """
Required class structure:
```python
from dataclasses import dataclass
from typing import Sequence
from fitz_ai.core.chunk import ChunkLike
from fitz_ai.core.guardrails.base import ConstraintResult

@dataclass
class MyConstraint:
    # Configuration parameters
    enabled: bool = True

    @property
    def name(self) -> str:
        return "my_constraint"

    def apply(
        self,
        query: str,
        chunks: Sequence[ChunkLike],
    ) -> ConstraintResult:
        '''
        Evaluate if chunks support answering the query.

        Args:
            query: User's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult.allow() if answer is safe
            ConstraintResult.deny(reason="...") if answer should be blocked
        '''
        pass
```

ConstraintResult:
- ConstraintResult.allow(): Allow decisive answer
- ConstraintResult.deny(reason="...", signal="..."): Block decisive answer
""",
    }

    return protocols.get(plugin_type, "Implement required plugin protocol.")


# =============================================================================
# Prompt Building
# =============================================================================


def build_generation_prompt(
    plugin_type: PluginType,
    description: str,
    example_code: Optional[str] = None,
) -> str:
    """
    Build the prompt for generating a plugin.

    Args:
        plugin_type: Type of plugin to generate
        description: User's description of what they want
        example_code: Optional example plugin code

    Returns:
        Complete prompt for the LLM
    """
    if example_code is None:
        example_code = load_example_plugin(plugin_type)

    # Build prompt parts
    parts = [
        f"Generate a {plugin_type.display_name} plugin for: {description}",
        "",
        "## Requirements",
        "",
    ]

    # Add schema or protocol info
    if plugin_type.is_yaml:
        parts.append(get_yaml_schema_info(plugin_type))
    else:
        parts.append(get_python_protocol_info(plugin_type))

    # Add expected behavior
    parts.extend(
        [
            "",
            "## Expected Behavior",
            "",
            get_expected_behavior(plugin_type),
        ]
    )

    # Add example if available
    if example_code:
        parts.extend(
            [
                "",
                "## Example Plugin (for reference)",
                "",
                f"```{'yaml' if plugin_type.is_yaml else 'python'}",
                example_code.strip(),
                "```",
            ]
        )

    # Add instructions
    extension = plugin_type.file_extension
    parts.extend(
        [
            "",
            "## Instructions",
            "",
            f"Generate a complete, working {extension} plugin.",
            "- Output ONLY the plugin code, no explanations",
            "- Ensure all required fields are present",
            "- Use realistic values based on the description",
            "- The plugin should work immediately after setting the required API key",
            "- IMPORTANT: Use only Python standard library or packages already imported in fitz-ai",
            "- Do NOT use nltk, spacy, or other NLP libraries that require extra downloads",
            "- For sentence splitting, use regex: re.split(r'(?<=[.!?])\\s+', text)",
            "",
            f"Output the complete {extension} code:",
        ]
    )

    return "\n".join(parts)


def build_retry_prompt(
    plugin_type: PluginType,
    description: str,
    previous_code: str,
    error_feedback: str,
) -> str:
    """
    Build a retry prompt after validation failure.

    Args:
        plugin_type: Type of plugin
        description: Original user description
        previous_code: The code that failed validation
        error_feedback: Formatted validation error

    Returns:
        Prompt for retry attempt
    """
    parts = [
        f"Your previous {plugin_type.display_name} plugin for '{description}' failed validation.",
        "",
        "## Error",
        "",
        error_feedback,
        "",
        "## Your Previous Code",
        "",
        f"```{'yaml' if plugin_type.is_yaml else 'python'}",
        previous_code.strip(),
        "```",
        "",
        "## Instructions",
        "",
        "Fix the error and regenerate the complete plugin.",
        "- Output ONLY the corrected plugin code",
        "- Ensure all required fields are present",
        "- Do not remove any necessary fields",
        "",
        "Output the corrected code:",
    ]

    return "\n".join(parts)


def extract_code_from_response(response: str, plugin_type: PluginType) -> str:
    """
    Extract code from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response
        plugin_type: Type of plugin (for extension)

    Returns:
        Extracted code string
    """
    response = response.strip()

    # Try to extract from markdown code blocks
    lang = "yaml" if plugin_type.is_yaml else "python"

    # Check for ```yaml or ```python blocks
    for marker in [f"```{lang}", "```"]:
        if marker in response:
            start = response.find(marker) + len(marker)
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

    # If no code block, return as-is (might be raw code)
    return response


__all__ = [
    "load_example_plugin",
    "get_yaml_schema_info",
    "get_python_protocol_info",
    "build_generation_prompt",
    "build_retry_prompt",
    "extract_code_from_response",
]
