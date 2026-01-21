# fitz_ai/cli/services/init_service.py
"""Service layer for init command - handles configuration generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SystemStatus:
    """System detection results."""

    api_keys: dict[str, Any]
    ollama: Any
    qdrant: Any
    faiss: Any


@dataclass
class ConfigResult:
    """Result of config generation."""

    global_config: str
    engine_config: str | None
    engine_name: str


class InitService:
    """
    Business logic for init command.

    Handles system detection, plugin filtering, and config generation
    without UI concerns. Can be used by CLI, SDK, or API.
    """

    def detect_system(self) -> SystemStatus:
        """Detect available services and API keys."""
        from fitz_ai.core.detect import detect_system_status

        system = detect_system_status()
        return SystemStatus(
            api_keys=system.api_keys,
            ollama=system.ollama,
            qdrant=system.qdrant,
            faiss=system.faiss,
        )

    def load_default_config(self) -> dict:
        """Load the default configuration from package."""
        defaults_path = (
            Path(__file__).parent.parent.parent / "engines" / "fitz_rag" / "config" / "default.yaml"
        )
        with defaults_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def filter_available_plugins(
        self, plugins: list[str], plugin_type: str, system: SystemStatus
    ) -> list[str]:
        """Filter plugins to only those that are available."""
        available = []

        for plugin in plugins:
            plugin_lower = plugin.lower()

            # Ollama plugins require Ollama
            if "ollama" in plugin_lower:
                if system.ollama.available:
                    available.append(plugin)
                continue

            # Qdrant requires Qdrant
            if "qdrant" in plugin_lower:
                if system.qdrant.available:
                    available.append(plugin)
                continue

            # FAISS requires faiss
            if "faiss" in plugin_lower:
                if system.faiss.available:
                    available.append(plugin)
                continue

            # API-based plugins require API keys
            if "cohere" in plugin_lower:
                if system.api_keys.get("cohere", type("", (), {"available": False})).available:
                    available.append(plugin)
                continue

            if "openai" in plugin_lower or "azure" in plugin_lower:
                if system.api_keys.get("openai", type("", (), {"available": False})).available:
                    available.append(plugin)
                continue

            if "anthropic" in plugin_lower:
                if system.api_keys.get("anthropic", type("", (), {"available": False})).available:
                    available.append(plugin)
                continue

            # Always available (local/fallback)
            if plugin_lower in ["simple", "markdown", "semantic", "docling"]:
                available.append(plugin)
                continue

        return available

    def get_default_model(self, plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
        """Get default model for a plugin."""
        from fitz_ai.llm.registry import get_llm_plugin

        try:
            instance = get_llm_plugin(plugin_type=plugin_type, plugin_name=plugin_name, tier=tier)
            return getattr(instance, "params", {}).get("model", "")
        except Exception:
            return ""

    def generate_global_config(self, default_engine: str) -> str:
        """Generate global config YAML."""
        config = {"runtime": {"default_engine": default_engine}}
        return yaml.dump(config, sort_keys=False, default_flow_style=False)

    def generate_fitz_rag_config(
        self,
        chat_plugin: str,
        chat_model: str,
        embedding_plugin: str,
        embedding_model: str,
        rerank_plugin: str | None,
        rerank_model: str | None,
        vector_db_plugin: str,
        retrieval_plugin: str,
        chunking_plugin: str,
        collection: str = "default",
    ) -> str:
        """Generate fitz_rag engine config."""
        config: dict[str, Any] = {
            "chat": {
                "plugin_name": chat_plugin,
                "kwargs": {},
            },
            "embedding": {
                "plugin_name": embedding_plugin,
                "kwargs": {},
            },
            "rerank": {
                "enabled": False,
                "plugin_name": None,
                "kwargs": {},
            },
            "vector_db": {
                "plugin_name": vector_db_plugin,
                "kwargs": {},
            },
            "retrieval": {
                "collection": collection,
                "plugin_name": retrieval_plugin,
                "top_k": 25,
                "fetch_artifacts": True,
            },
            "chunking": {
                "router": {
                    "default": {
                        "plugin_name": chunking_plugin,
                        "kwargs": {
                            "chunk_size": 1500,
                            "chunk_overlap": 200,
                        },
                    },
                },
            },
            "rgs": {
                "enable_citations": True,
                "strict_grounding": True,
                "answer_style": "concise",
                "max_chunks": 10,
            },
        }

        # Add model overrides if provided
        if chat_model:
            config["chat"]["kwargs"]["model"] = chat_model
        if embedding_model:
            config["embedding"]["kwargs"]["model"] = embedding_model

        # Add rerank if provided
        if rerank_plugin and rerank_model:
            config["rerank"]["enabled"] = True
            config["rerank"]["plugin_name"] = rerank_plugin
            config["rerank"]["kwargs"]["model"] = rerank_model

        return yaml.dump(config, sort_keys=False, default_flow_style=False)

    def copy_engine_default_config(self, engine_name: str, registry: Any) -> str | None:
        """Copy engine's default config if it exists."""
        engine_meta = registry.get_metadata(engine_name)
        if not engine_meta or not engine_meta.config_path:
            return None

        config_path = Path(engine_meta.config_path)
        if not config_path.exists():
            return None

        with config_path.open("r", encoding="utf-8") as f:
            return f.read()

    def generate_graphrag_config(
        self,
        llm_provider: str,
        embedding_provider: str,
        storage_backend: str = "memory",
    ) -> str:
        """Generate GraphRAG engine config."""
        config = {
            "llm": {
                "provider": llm_provider,
                "model": "gpt-4o-mini" if llm_provider == "openai" else "command-r",
            },
            "embeddings": {
                "provider": embedding_provider,
                "model": "text-embedding-3-small"
                if embedding_provider == "openai"
                else "embed-english-v3.0",
            },
            "storage": {
                "type": storage_backend,
            },
        }
        return yaml.dump(config, sort_keys=False, default_flow_style=False)

    def generate_clara_config(
        self,
        variant: str = "e2e",
        device: str = "cuda",
        compression_rate: int = 16,
    ) -> str:
        """Generate Clara engine config."""
        config = {
            "variant": variant,
            "device": device,
            "compression_rate": compression_rate,
        }
        return yaml.dump(config, sort_keys=False, default_flow_style=False)

    def write_config(self, global_config: str, engine_config: str | None, engine_name: str) -> None:
        """Write configuration files to disk."""
        from fitz_ai.core.paths import FitzPaths

        paths = FitzPaths()

        # Write global config
        paths.global_config.parent.mkdir(parents=True, exist_ok=True)
        paths.global_config.write_text(global_config, encoding="utf-8")

        # Write engine config if provided
        if engine_config:
            engine_config_path = paths.config_dir / f"{engine_name}.yaml"
            engine_config_path.write_text(engine_config, encoding="utf-8")

    def validate_config(self, engine_name: str) -> bool:
        """Validate generated configuration by attempting to load it."""
        try:
            from fitz_ai.runtime import get_engine_registry

            registry = get_engine_registry()
            registry.create_engine(engine_name)
            return True
        except Exception:
            return False
