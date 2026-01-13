# tests/e2e/runner.py
"""
E2E test runner with ingestion and cleanup.

Handles the full lifecycle:
1. Create a unique test collection
2. Ingest test fixtures
3. Build RAG pipeline
4. Run test scenarios
5. Clean up collection and associated files
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fitz_ai.logging.logger import get_logger

from .scenarios import SCENARIOS, TestScenario
from .validators import ValidationResult, validate_answer

logger = get_logger(__name__)

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""

    scenario: TestScenario
    validation: ValidationResult
    answer_text: str
    duration_ms: float
    error: Optional[str] = None


@dataclass
class E2ERunResult:
    """Result of running all E2E tests."""

    collection: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    ingestion_duration_s: float = 0.0
    total_duration_s: float = 0.0

    @property
    def passed(self) -> int:
        """Number of passed scenarios."""
        return sum(1 for r in self.scenario_results if r.validation.passed)

    @property
    def failed(self) -> int:
        """Number of failed scenarios."""
        return sum(1 for r in self.scenario_results if not r.validation.passed)

    @property
    def total(self) -> int:
        """Total number of scenarios."""
        return len(self.scenario_results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


class E2ERunner:
    """
    End-to-end test runner.

    Usage:
        runner = E2ERunner()
        runner.setup()  # Ingests fixtures
        results = runner.run_all()  # Runs all scenarios
        runner.teardown()  # Cleans up collection
    """

    def __init__(
        self,
        fixtures_dir: Path | None = None,
        collection_prefix: str = "e2e_test",
    ):
        """
        Initialize the E2E runner.

        Args:
            fixtures_dir: Path to test fixtures (default: tests/e2e/fixtures)
            collection_prefix: Prefix for test collection name
        """
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.collection = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self.pipeline = None
        self.vector_client = None
        self._setup_complete = False

    def setup(self) -> float:
        """
        Set up the test environment.

        Creates collection, ingests fixtures, builds pipeline.
        Uses plugins configured in fitz.yaml (via CLIContext).

        Returns:
            Duration of ingestion in seconds
        """
        from fitz_ai.cli.context import CLIContext
        from fitz_ai.engines.fitz_rag.config import FitzRagConfig
        from fitz_ai.engines.fitz_rag.config.schema import (
            ChunkingRouterConfig,
            ExtensionChunkerConfig,
        )
        from fitz_ai.engines.fitz_rag.pipeline.engine import RAGPipeline
        from fitz_ai.ingestion.chunking.router import ChunkingRouter
        from fitz_ai.ingestion.diff import run_diff_ingest
        from fitz_ai.ingestion.parser import ParserRouter
        from fitz_ai.ingestion.state import IngestStateManager
        from fitz_ai.llm.registry import get_llm_plugin
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        logger.info(f"E2E Setup: Creating collection '{self.collection}'")

        # Clean up any stale test data from previous interrupted runs
        # This ensures a clean slate even if teardown didn't run
        self._cleanup_stale_data()

        start_time = time.time()

        # Load config from fitz.yaml via CLIContext
        ctx = CLIContext.load()
        config = ctx.raw_config

        # DEBUG: Print config source
        logger.info(f"E2E Setup: Config source: {ctx.config_source}")
        logger.info(f"E2E Setup: Config path: {ctx.config_path}")
        logger.info(f"E2E Setup: Has user config: {ctx.has_user_config}")

        # Get plugin names from config
        chat_plugin = config.get("chat", {}).get("plugin_name", "openai")
        chat_kwargs = config.get("chat", {}).get("kwargs", {})
        embedding_plugin = config.get("embedding", {}).get("plugin_name", "openai")
        embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
        vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
        vector_db_kwargs = config.get("vector_db", {}).get("kwargs", {})

        logger.info(
            f"E2E Setup: Using chat={chat_plugin}, embedding={embedding_plugin}, vector_db={vector_db_plugin}"
        )

        # Initialize components for ingestion
        self.vector_client = get_vector_db_plugin(vector_db_plugin, **vector_db_kwargs)

        # Embedder
        embedder = get_llm_plugin(
            plugin_type="embedding",
            plugin_name=embedding_plugin,
            **embedding_kwargs,
        )

        # Parser and chunking - use simple chunker for text
        parser_router = ParserRouter(docling_parser="docling")
        simple_chunker = ExtensionChunkerConfig(
            plugin_name="simple",
            # Larger chunks to keep product sections/facts together
            kwargs={"chunk_size": 2000, "chunk_overlap": 200},
        )
        router_config = ChunkingRouterConfig(
            default=simple_chunker,
            by_extension={
                ".md": simple_chunker,
                ".py": simple_chunker,
                ".txt": simple_chunker,
            },
        )
        chunking_router = ChunkingRouter.from_config(router_config)

        # State manager
        state_manager = IngestStateManager()
        state_manager.load()

        # Vector DB writer adapter
        class VectorDBWriterAdapter:
            def __init__(self, client):
                self._client = client

            def upsert(self, collection: str, points: list, defer_persist: bool = False):
                self._client.upsert(collection, points, defer_persist=defer_persist)

            def flush(self):
                if hasattr(self._client, "flush"):
                    self._client.flush()

        writer = VectorDBWriterAdapter(self.vector_client)

        # Skip enrichment for faster E2E tests (enrichment is tested separately)
        enrichment_pipeline = None

        # Run ingestion
        logger.info(f"E2E Setup: Ingesting fixtures from '{self.fixtures_dir}'")

        summary = run_diff_ingest(
            source=str(self.fixtures_dir),
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser_router=parser_router,
            chunking_router=chunking_router,
            collection=self.collection,
            embedding_id=embedding_plugin,
            vector_db_id=vector_db_plugin,
            enrichment_pipeline=enrichment_pipeline,
            force=True,  # Always re-ingest for clean test
        )

        ingestion_duration = time.time() - start_time
        logger.info(
            f"E2E Setup: Ingested {summary.ingested} files, "
            f"{summary.hierarchy_summaries} summaries in {ingestion_duration:.1f}s"
        )

        # Build RAG pipeline using configured plugins
        logger.info("E2E Setup: Building RAG pipeline")

        config_dict = {
            "chat": {"plugin_name": chat_plugin, "kwargs": chat_kwargs},
            "embedding": {"plugin_name": embedding_plugin, "kwargs": embedding_kwargs},
            "vector_db": {"plugin_name": vector_db_plugin, "kwargs": vector_db_kwargs},
            "retrieval": {
                "plugin_name": "dense",
                "collection": self.collection,
                "top_k": 20,  # Higher for aggregation queries (20 * 3x = 60 chunks)
            },
            "multihop": {"max_hops": 2},
            "rgs": {
                "strict_grounding": False,  # Allow more flexible answers in tests
                "max_chunks": 50,  # Higher limit to accommodate aggregation queries (3x multiplier)
            },
        }

        cfg = FitzRagConfig.from_dict(config_dict)
        # Disable default constraints for E2E tests to isolate retrieval testing
        # (constraints=[] prevents InsufficientEvidence/ConflictAware from blocking)
        # Also disable keyword matching - auto-detected vocabulary is too aggressive
        # for small test fixtures and filters out valid results
        self.pipeline = RAGPipeline.from_config(cfg, constraints=[], enable_keywords=False)

        self._setup_complete = True
        logger.info("E2E Setup: Complete")

        return ingestion_duration

    def _cleanup_stale_data(self) -> None:
        """
        Clean up any stale data from previous test runs.

        Called at the start of setup() to ensure a clean slate, even if
        a previous test run was interrupted and teardown didn't execute.
        """
        from fitz_ai.cli.context import CLIContext
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        try:
            # Need vector client to delete collection
            ctx = CLIContext.load()
            config = ctx.raw_config
            vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
            vector_db_kwargs = config.get("vector_db", {}).get("kwargs", {})

            temp_client = get_vector_db_plugin(vector_db_plugin, **vector_db_kwargs)

            # Try to delete the collection (fails silently if doesn't exist)
            try:
                temp_client.delete_collection(self.collection)
                logger.debug(f"E2E Setup: Cleaned up stale collection '{self.collection}'")
            except Exception:
                pass  # Collection didn't exist, which is fine

            # Try to delete table collection
            try:
                temp_client.delete_collection(f"{self.collection}_tables")
                logger.debug(f"E2E Setup: Cleaned up stale table collection")
            except Exception:
                pass

            # Delete associated files
            self._delete_vocabulary()
            self._delete_table_registry()
            self._delete_entity_graph()
            self._delete_table_store()

        except Exception as e:
            logger.debug(f"E2E Setup: Stale data cleanup encountered error (non-fatal): {e}")

    def teardown(self) -> None:
        """
        Clean up the test environment.

        Deletes the test collection and associated files.
        """
        if not self.vector_client:
            return

        logger.info(f"E2E Teardown: Deleting collection '{self.collection}'")

        try:
            # Delete vector collection
            deleted = self.vector_client.delete_collection(self.collection)
            logger.info(f"E2E Teardown: Deleted {deleted} vectors")
        except Exception as e:
            logger.warning(f"E2E Teardown: Failed to delete collection: {e}")

        # Delete associated files
        self._delete_vocabulary()
        self._delete_table_registry()
        self._delete_entity_graph()
        self._delete_table_store()

        self._setup_complete = False
        logger.info("E2E Teardown: Complete")

    def _delete_vocabulary(self) -> None:
        """Delete vocabulary file associated with collection."""
        from fitz_ai.core.paths import FitzPaths

        vocab_path = FitzPaths.vocabulary(self.collection)
        if vocab_path.exists():
            try:
                vocab_path.unlink()
                logger.debug(f"Deleted vocabulary: {vocab_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete vocabulary: {e}")

    def _delete_table_registry(self) -> None:
        """Delete table registry file associated with collection."""
        from fitz_ai.core.paths import FitzPaths

        registry_path = FitzPaths.table_registry(self.collection)
        if registry_path.exists():
            try:
                registry_path.unlink()
                logger.debug(f"Deleted table registry: {registry_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete table registry: {e}")

    def _delete_entity_graph(self) -> None:
        """Delete entity graph database associated with collection."""
        from fitz_ai.core.paths import FitzPaths

        graph_path = FitzPaths.entity_graph(self.collection)
        if graph_path.exists():
            try:
                graph_path.unlink()
                logger.debug(f"Deleted entity graph: {graph_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete entity graph: {e}")

    def _delete_table_store(self) -> None:
        """Delete table storage (both collection and cache) associated with collection."""
        try:
            # Delete the table collection in vector DB (e.g., {collection}_tables for Qdrant)
            table_collection = f"{self.collection}_tables"
            try:
                self.vector_client.delete_collection(table_collection)
                logger.debug(f"Deleted table collection: {table_collection}")
            except Exception:
                # Collection might not exist
                pass

            # Delete SQLite cache used by GenericTableStore
            from fitz_ai.core.paths import FitzPaths

            cache_path = FitzPaths.workspace() / f"tables_{self.collection}.db"
            if cache_path.exists():
                try:
                    cache_path.unlink()
                    logger.debug(f"Deleted table cache: {cache_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete table cache: {e}")
        except Exception as e:
            logger.warning(f"Failed to delete table store: {e}")

    def run_scenario(self, scenario: TestScenario) -> ScenarioResult:
        """
        Run a single test scenario.

        Args:
            scenario: The scenario to run

        Returns:
            ScenarioResult with validation outcome
        """
        if not self._setup_complete:
            raise RuntimeError("E2E runner not set up. Call setup() first.")

        logger.debug(f"Running scenario {scenario.id}: {scenario.name}")

        start_time = time.time()
        error = None
        answer_text = ""

        try:
            answer = self.pipeline.run(scenario.query)
            answer_text = answer.answer
            validation = validate_answer(answer, scenario)
        except Exception as e:
            logger.error(f"Scenario {scenario.id} failed with error: {e}")
            error = str(e)
            validation = ValidationResult(
                passed=False,
                reason=f"Pipeline error: {e}",
                details={"error": str(e)},
            )

        duration_ms = (time.time() - start_time) * 1000

        return ScenarioResult(
            scenario=scenario,
            validation=validation,
            answer_text=answer_text[:500] if answer_text else "",
            duration_ms=duration_ms,
            error=error,
        )

    def run_all(self, scenarios: list[TestScenario] | None = None) -> E2ERunResult:
        """
        Run all test scenarios.

        Args:
            scenarios: Optional list of scenarios (default: all SCENARIOS)

        Returns:
            E2ERunResult with all scenario results
        """
        scenarios = scenarios or SCENARIOS
        results: list[ScenarioResult] = []

        logger.info(f"E2E Run: Starting {len(scenarios)} scenarios")
        start_time = time.time()

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

            status = "PASS" if result.validation.passed else "FAIL"
            logger.info(f"  [{status}] {scenario.id}: {scenario.name} ({result.duration_ms:.0f}ms)")

        total_duration = time.time() - start_time

        run_result = E2ERunResult(
            collection=self.collection,
            scenario_results=results,
            total_duration_s=total_duration,
        )

        logger.info(
            f"E2E Run: Complete - {run_result.passed}/{run_result.total} passed "
            f"({run_result.pass_rate:.1f}%) in {total_duration:.1f}s"
        )

        return run_result


def run_e2e_tests(
    fixtures_dir: Path | None = None,
    scenarios: list[TestScenario] | None = None,
) -> E2ERunResult:
    """
    Convenience function to run E2E tests.

    Handles setup, execution, and teardown automatically.

    Args:
        fixtures_dir: Path to fixtures (default: tests/e2e/fixtures)
        scenarios: Optional specific scenarios to run

    Returns:
        E2ERunResult with all outcomes
    """
    runner = E2ERunner(fixtures_dir=fixtures_dir)

    try:
        ingestion_duration = runner.setup()
        result = runner.run_all(scenarios)
        result.ingestion_duration_s = ingestion_duration
        return result
    finally:
        runner.teardown()


if __name__ == "__main__":
    # Run E2E tests from command line
    import sys

    print("=" * 60)
    print("E2E Retrieval Intelligence Tests")
    print("=" * 60)
    print()

    result = run_e2e_tests()

    print()
    print("=" * 60)
    print(f"Results: {result.passed}/{result.total} passed ({result.pass_rate:.1f}%)")
    print(f"Ingestion: {result.ingestion_duration_s:.1f}s")
    print(f"Total: {result.total_duration_s:.1f}s")
    print("=" * 60)

    # Print failed scenarios
    failed = [r for r in result.scenario_results if not r.validation.passed]
    if failed:
        print()
        print("Failed Scenarios:")
        for r in failed:
            print(f"  [{r.scenario.id}] {r.scenario.name}")
            print(f"      Reason: {r.validation.reason}")
            print()

    sys.exit(0 if result.failed == 0 else 1)
