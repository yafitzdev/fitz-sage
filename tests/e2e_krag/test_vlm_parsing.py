# tests/e2e_krag/test_vlm_parsing.py
"""
KRAG E2E tests for VLM-powered figure parsing via Ollama.

Tests that DoclingVisionParser + OllamaVision correctly extract figure
descriptions from PDFs, and that KRAG retrieval can answer questions
about chart/figure data.

Requires:
- Ollama running locally (http://localhost:11434)
- llava:7b model pulled (ollama pull llava:7b)

Run with: pytest tests/e2e_krag/test_vlm_parsing.py -v -m e2e_krag_parser

Skip behavior:
- Skips entire module if Ollama is not reachable
- Skips entire module if llava:7b model is not available
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e_krag.scenarios import Feature, TestScenario

# Mark all tests in this module
pytestmark = [pytest.mark.e2e_krag_parser, pytest.mark.llm]

# Fixtures directory with figure_test.pdf
FIXTURES_PARSER_DIR = Path(__file__).parent / "fixtures_parser"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_VISION_MODEL = "llava:7b"


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def _model_available(model: str) -> bool:
    """Check if a specific model is pulled in Ollama."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        data = resp.json()
        model_names = [m.get("name", "") for m in data.get("models", [])]
        # Match with or without tag suffix
        return any(
            model in name or name.startswith(model.split(":")[0] + ":") for name in model_names
        )
    except Exception:
        return False


# Skip entire module if Ollama or model not available
pytestmark.append(
    pytest.mark.skipif(
        not _ollama_available(),
        reason="Ollama not reachable at localhost:11434",
    )
)
pytestmark.append(
    pytest.mark.skipif(
        _ollama_available() and not _model_available(OLLAMA_VISION_MODEL),
        reason=f"Ollama model {OLLAMA_VISION_MODEL} not available (run: ollama pull {OLLAMA_VISION_MODEL})",
    )
)


# VLM-specific figure retrieval scenarios
VLM_SCENARIOS: list[TestScenario] = [
    TestScenario(
        id="E145",
        name="Figure: chart data retrieval",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the projected quantum computing market size in 2028?",
        must_contain_any=["42.7", "billion", "2028"],
        min_sources=1,
    ),
    TestScenario(
        id="E146",
        name="Figure: caption information",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the CAGR for the quantum computing market shown in the figure?",
        must_contain_any=["72.3%", "72.3", "CAGR"],
        min_sources=1,
    ),
    TestScenario(
        id="E147",
        name="Figure: market growth context",
        feature=Feature.FIGURE_RETRIEVAL,
        query=(
            "According to the market analysis figure, what was the quantum "
            "computing market size in 2024?"
        ),
        must_contain_any=["4.8", "billion", "2024"],
        min_sources=1,
    ),
]


@pytest.fixture(scope="module")
def vlm_krag_engine(set_workspace):
    """
    Module-scoped KRAG engine configured with OllamaVision for VLM parsing.

    Creates a unique collection, ingests fixtures_parser/ with docling_vision
    parser and Ollama vision provider, then yields the engine for querying.
    """
    from fitz_ai.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_ai.engines.fitz_krag.engine import FitzKragEngine
    from fitz_ai.storage.postgres import PostgresConnectionManager
    from tests.e2e_krag.config import get_tier_config, get_tier_names, load_e2e_config

    collection = f"e2e_vlm_{uuid.uuid4().hex[:8]}"

    # Load base tier config for chat/embedding/vector_db
    e2e_config = load_e2e_config()
    tier_names = get_tier_names(e2e_config)
    tier_config = get_tier_config(tier_names[0], e2e_config)

    config_dict = {
        "chat": tier_config["chat"]["plugin_name"],
        "embedding": tier_config["embedding"]["plugin_name"],
        "vector_db": tier_config["vector_db"]["plugin_name"],
        "collection": collection,
        # VLM config
        "vision": f"ollama/{OLLAMA_VISION_MODEL}",
        "parser": "docling_vision",
        # Relax for testing
        "enable_guardrails": False,
        "strict_grounding": False,
        "top_addresses": 20,
        "top_read": 10,
        # Plugin kwargs from tier config
        "chat_kwargs": tier_config["chat"].get("kwargs", {}),
        "embedding_kwargs": tier_config["embedding"].get("kwargs", {}),
        "vector_db_kwargs": tier_config["vector_db"].get("kwargs", {}),
    }

    cfg = FitzKragConfig(**config_dict)
    engine = FitzKragEngine(cfg)

    # Ingest fixtures with VLM-powered parsing via KragIngestPipeline
    from fitz_ai.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    pipeline = KragIngestPipeline(
        config=engine._config,
        chat=engine._chat,
        embedder=engine._embedder,
        connection_manager=engine._connection_manager,
        collection=engine._config.collection,
        table_store=engine._table_store,
        pg_table_store=engine._pg_table_store,
        vocabulary_store=engine._vocabulary_store,
        entity_graph_store=engine._entity_graph_store,
    )
    stats = pipeline.ingest(FIXTURES_PARSER_DIR, force=True)
    print(
        f"\nVLM KRAG ingest: {stats.get('files_scanned', 0)} files, "
        f"{stats.get('sections_extracted', 0)} sections"
    )

    yield engine

    # Cleanup
    try:
        conn_mgr = PostgresConnectionManager.get_instance()
        for table in [
            "krag_raw_files",
            "krag_symbol_index",
            "krag_import_graph",
            "krag_section_index",
            "krag_table_index",
        ]:
            try:
                conn_mgr.execute(collection, f'DROP TABLE IF EXISTS "{table}" CASCADE')
            except Exception:
                pass
    except Exception:
        pass


@pytest.mark.parametrize(
    "scenario",
    VLM_SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_vlm_figure_scenario(vlm_krag_engine, scenario):
    """
    Run a VLM figure retrieval scenario through KRAG engine.

    Validates that VLM-described figure content is retrievable and
    contains expected data values.
    """
    from fitz_ai.core import Query

    from .validators import validate_answer

    start = time.time()
    answer = vlm_krag_engine.answer(Query(text=scenario.query))
    duration_ms = (time.time() - start) * 1000

    validation = validate_answer(answer, scenario)

    if not validation.passed:
        pytest.fail(
            f"\n\nVLM Scenario {scenario.id} ({scenario.name}) FAILED\n"
            f"Feature: {scenario.feature.value}\n"
            f"Query: {scenario.query}\n"
            f"Reason: {validation.reason}\n"
            f"Details: {validation.details}\n"
            f"Answer preview: {answer.text[:300] if answer.text else '(no answer)'}...\n"
            f"Duration: {duration_ms:.0f}ms"
        )
