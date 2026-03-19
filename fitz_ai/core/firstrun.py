# fitz_ai/core/firstrun.py
"""
First-run experience for fitz-ai.

Handles auto-detection of LLM providers and interactive setup when no
config exists. Called by CLI before engine creation on first run.

Flow:
    1. Check if .fitz/config.yaml exists → if yes, skip
    2. Detect Ollama → list models → classify into tiers
    3. If models missing → prompt user to pull them
    4. Fallback: API keys → helpful error
    5. Write config and continue
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from fitz_ai.core.paths import FitzPaths

logger = logging.getLogger(__name__)

# Embedding model families (name patterns that indicate embedding models)
_EMBEDDING_PATTERNS = re.compile(r"embed|nomic|bge|mxbai-embed|e5-", re.IGNORECASE)

# Recommended lightweight models for first-run
RECOMMENDED_CHAT = "qwen3.5:0.6b"
RECOMMENDED_EMBEDDING = "nomic-embed-text"


@dataclass
class OllamaModel:
    """A model available in Ollama."""

    name: str
    size_bytes: int = 0
    parameter_size: str = ""
    family: str = ""
    is_embedding: bool = False


@dataclass
class DetectedModels:
    """Result of Ollama model detection."""

    chat_models: list[OllamaModel] = field(default_factory=list)
    embedding_models: list[OllamaModel] = field(default_factory=list)


def _ollama_binary_exists() -> bool:
    """Check if the ollama binary is on PATH (installed but maybe not running)."""
    import shutil

    return shutil.which("ollama") is not None


def needs_firstrun() -> bool:
    """Check if first-run setup is needed (no config exists)."""
    return not FitzPaths.config().exists()


def list_ollama_models() -> list[OllamaModel] | None:
    """
    List all models available in Ollama via /api/tags.

    Returns None if Ollama is not running.
    """
    try:
        import httpx

        from fitz_ai.core.constants import (
            OLLAMA_API_TAGS_PATH,
            OLLAMA_DEFAULT_PORT,
            OLLAMA_HEALTH_TIMEOUT,
        )
    except ImportError:
        return None

    for host in ["localhost", "127.0.0.1"]:
        try:
            response = httpx.get(
                f"http://{host}:{OLLAMA_DEFAULT_PORT}{OLLAMA_API_TAGS_PATH}",
                timeout=OLLAMA_HEALTH_TIMEOUT,
            )
            if response.status_code != 200:
                continue

            data = response.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                details = m.get("details", {})
                models.append(
                    OllamaModel(
                        name=name,
                        size_bytes=m.get("size", 0),
                        parameter_size=details.get("parameter_size", ""),
                        family=details.get("family", ""),
                        is_embedding=bool(_EMBEDDING_PATTERNS.search(name)),
                    )
                )
            return models
        except Exception:
            continue

    return None


def classify_models(models: list[OllamaModel]) -> DetectedModels:
    """Classify Ollama models into chat and embedding categories."""
    result = DetectedModels()
    for m in models:
        if m.is_embedding:
            result.embedding_models.append(m)
        else:
            result.chat_models.append(m)
    return result


def _parse_param_size(size_str: str) -> float:
    """Parse parameter size string like '3B', '14B', '0.6B' to float."""
    match = re.search(r"([\d.]+)\s*[bB]", size_str)
    if match:
        return float(match.group(1))
    return 0.0


def assign_tiers(chat_models: list[OllamaModel]) -> dict[str, str]:
    """
    Assign chat models to fast/balanced/smart tiers by parameter size.

    If only one model, use it for all tiers.
    """
    if not chat_models:
        return {}

    # Sort by parameter size
    sorted_models = sorted(chat_models, key=lambda m: _parse_param_size(m.parameter_size))

    if len(sorted_models) == 1:
        name = sorted_models[0].name
        return {"fast": name, "balanced": name, "smart": name}

    if len(sorted_models) == 2:
        return {
            "fast": sorted_models[0].name,
            "balanced": sorted_models[1].name,
            "smart": sorted_models[1].name,
        }

    return {
        "fast": sorted_models[0].name,
        "balanced": sorted_models[len(sorted_models) // 2].name,
        "smart": sorted_models[-1].name,
    }


def write_config(
    chat_fast: str,
    chat_balanced: str,
    chat_smart: str,
    embedding: str,
    rerank: str | None = None,
) -> Path:
    """Write the .fitz/config.yaml file."""
    config_path = FitzPaths.config()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    rerank_line = f"rerank: {rerank}" if rerank else "# rerank: cohere/rerank-v3.5"
    content = f"""\
# Fitz Configuration
# Docs: https://github.com/yafitzdev/fitz-ai/blob/main/docs/CONFIG.md

# Chat models by tier (provider/model)
chat_fast: {chat_fast}
chat_balanced: {chat_balanced}
chat_smart: {chat_smart}

# Embedding model
embedding: {embedding}

# Optional (uncomment to enable)
{rerank_line}
# vision: ollama/llava

collection: default
"""
    config_path.write_text(content, encoding="utf-8")
    return config_path


def pull_ollama_model(model: str) -> bool:
    """Pull an Ollama model. Returns True on success."""
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=False,
            timeout=600,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Failed to pull {model}: {e}")
        return False


def run_firstrun_setup() -> bool:
    """
    Interactive first-run setup. Returns True if config was written.

    Detects available providers and models, prompts user if needed,
    writes .fitz/config.yaml.
    """
    config_path = FitzPaths.config()

    # ── Try Ollama ──────────────────────────────────────────────
    models = list_ollama_models()

    if models is not None:
        detected = classify_models(models)
        tiers = assign_tiers(detected.chat_models)
        has_embedding = len(detected.embedding_models) > 0

        if tiers and has_embedding:
            # Everything available — auto-configure
            embedding_name = detected.embedding_models[0].name
            write_config(
                chat_fast=f"ollama/{tiers['fast']}",
                chat_balanced=f"ollama/{tiers['balanced']}",
                chat_smart=f"ollama/{tiers['smart']}",
                embedding=f"ollama/{embedding_name}",
            )
            print("\n  Auto-configured from Ollama models:")
            print(f"    fast:      {tiers['fast']}")
            print(f"    balanced:  {tiers['balanced']}")
            print(f"    smart:     {tiers['smart']}")
            print(f"    embedding: {embedding_name}")
            print(f"\n  Config: {config_path}")
            print("  Edit this file to change models.\n")
            return True

        # Missing models — prompt to pull
        missing = []
        if not tiers:
            missing.append(("chat", RECOMMENDED_CHAT))
        if not has_embedding:
            missing.append(("embedding", RECOMMENDED_EMBEDDING))

        print("\n  Ollama is running but missing required models.\n")

        if not tiers and detected.chat_models:
            pass  # Shouldn't happen, but just in case
        elif not tiers:
            print("  Chat model needed. Example:")
            print(f"    ollama pull {RECOMMENDED_CHAT}\n")
        if not has_embedding:
            print("  Embedding model needed. Example:")
            print(f"    ollama pull {RECOMMENDED_EMBEDDING}\n")

        # Offer to pull
        missing_names = [m[1] for m in missing]
        total_desc = " + ".join(missing_names)
        try:
            answer = input(f"  Pull {total_desc}? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("", "y", "yes"):
            for _kind, model_name in missing:
                print(f"  Pulling {model_name}...")
                if not pull_ollama_model(model_name):
                    print(f"  Failed to pull {model_name}.")
                    print(f"  Run manually: ollama pull {model_name}")
                    print(f"\n  Config: {config_path}\n")
                    return False

            # Re-detect after pulling
            models = list_ollama_models()
            if models:
                detected = classify_models(models)
                tiers = assign_tiers(detected.chat_models)
                if tiers and detected.embedding_models:
                    embedding_name = detected.embedding_models[0].name
                    write_config(
                        chat_fast=f"ollama/{tiers['fast']}",
                        chat_balanced=f"ollama/{tiers['balanced']}",
                        chat_smart=f"ollama/{tiers['smart']}",
                        embedding=f"ollama/{embedding_name}",
                    )
                    print(f"\n  Configured. Config: {config_path}\n")
                    return True

            print("  Could not configure after pulling. Edit config manually.")
            print(f"  Config: {config_path}\n")
            return False
        else:
            print("\n  Pull the models manually, then run fitz again:")
            for _kind, model_name in missing:
                print(f"    ollama pull {model_name}")
            print(f"\n  Config: {config_path}\n")
            return False

    # ── Ollama installed but not running? ──────────────────────
    if models is None and _ollama_binary_exists():
        print("\n  Ollama is installed but not running.")
        print("  Start it with: ollama serve")
        print("\n  Then run fitz again.\n")
        return False

    # ── Try API keys ────────────────────────────────────────────
    import os

    for provider, env_var in [
        ("cohere", "COHERE_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ]:
        if os.getenv(env_var):
            write_config(
                chat_fast=provider,
                chat_balanced=provider,
                chat_smart=provider,
                embedding=provider,
                rerank="cohere/rerank-v3.5" if provider == "cohere" else None,
            )
            print(f"\n  Configured with {provider} ({env_var} detected).")
            print(f"  Config: {config_path}\n")
            return True

    # ── Nothing available ───────────────────────────────────────
    print("\n  No LLM provider found. Set up one of these:\n")
    print("  Option 1 — Ollama (local, free):")
    print("    Install from https://ollama.ai")
    print(f"    ollama pull {RECOMMENDED_CHAT}")
    print(f"    ollama pull {RECOMMENDED_EMBEDDING}\n")
    print("  Option 2 — Cohere (cloud, free tier):")
    print("    export COHERE_API_KEY=your-key-here\n")
    print("  Option 3 — OpenAI:")
    print("    export OPENAI_API_KEY=your-key-here\n")
    print("  Then run fitz again.\n")
    print(f"  Config: {config_path}\n")
    return False
