# tests/e2e/runner.py
"""
E2E test runner with ingestion and cleanup.

Handles the full lifecycle:
1. Create a unique test collection
2. Ingest test fixtures
3. Build RAG pipeline
4. Run test scenarios (with tiered fallback)
5. Clean up collection and associated files
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fitz_ai.logging.logger import get_logger

from .cache import ResponseCache
from .config import get_cache_config, get_tier_config, get_tier_names, load_e2e_config
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
    chunk_ids: list[str] = field(default_factory=list)


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


@dataclass
class TieredRunResult:
    """Result of tiered test execution."""

    results: dict[str, tuple[ScenarioResult, str]] = field(
        default_factory=dict
    )  # id -> (result, tier)
    tier_names: list[str] = field(default_factory=list)
    cache_stats: dict = field(default_factory=dict)
    total_duration_s: float = 0.0

    def tier_summary(self) -> dict[str, int]:
        """Count scenarios by tier."""
        counts: dict[str, int] = {}
        for _result, tier in self.results.values():
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    @property
    def total_passed(self) -> int:
        """Total scenarios that passed (any tier)."""
        return sum(1 for r, _t in self.results.values() if r.validation.passed)

    @property
    def total_failed(self) -> int:
        """Total scenarios that failed all tiers."""
        return sum(1 for r, _t in self.results.values() if not r.validation.passed)

    @property
    def total(self) -> int:
        """Total scenarios."""
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        return (self.total_passed / self.total * 100) if self.total > 0 else 0.0

    def print_summary(self) -> None:
        """Print a summary of tiered results."""
        print("\n" + "=" * 60)
        print("TIERED E2E TEST RESULTS")
        print("=" * 60)

        tier_counts = self.tier_summary()
        cached_count = tier_counts.get("cached", 0)
        failed_all_count = tier_counts.get("failed_all", 0)

        # Show cache hits first
        if cached_count > 0:
            print(f"  Cached (skipped LLM):  {cached_count}")

        # Show each tier's passes (always show all tiers for clarity)
        for tier in self.tier_names:
            count = tier_counts.get(tier, 0)
            print(f"  {tier.capitalize()} tier passed:   {count}")

        # Show failures with explicit tier info
        if failed_all_count > 0:
            tiers_tried = " -> ".join(self.tier_names) if self.tier_names else "none"
            print(f"  Failed ALL tiers:      {failed_all_count}  (tried: {tiers_tried})")

        print("-" * 60)
        print(f"  Total: {self.total_passed}/{self.total} passed ({self.pass_rate:.1f}%)")
        print(f"  Duration: {self.total_duration_s:.1f}s")

        if self.cache_stats:
            print(
                f"  Cache: {self.cache_stats.get('hits', 0)} hits, {self.cache_stats.get('misses', 0)} misses"
            )
        print("=" * 60)


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
        use_cache: bool = True,
    ):
        """
        Initialize the E2E runner.

        Args:
            fixtures_dir: Path to test fixtures (default: tests/e2e/fixtures)
            collection_prefix: Prefix for test collection name
            use_cache: Whether to use response caching
        """
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self.collection = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self.pipeline = None
        self.vector_client = None
        self._setup_complete = False
        self._current_tier: str | None = None
        self._tiered_results: TieredRunResult | None = None

        # Initialize cache
        cache_config = get_cache_config()
        self.cache = ResponseCache(
            max_entries=cache_config.get("max_entries", 1000),
            ttl_days=cache_config.get("ttl_days", 30),
            enabled=use_cache and cache_config.get("enabled", True),
        )

    def get_tiered_result(self, scenario_id: str) -> tuple[ScenarioResult, str] | None:
        """
        Get pre-computed result for a scenario from tiered execution.

        Args:
            scenario_id: The scenario ID to look up

        Returns:
            Tuple of (ScenarioResult, tier_name) if tiered results exist, None otherwise
        """
        if self._tiered_results is None:
            return None
        return self._tiered_results.results.get(scenario_id)

    def setup(self, tier_name: str | None = None) -> float:
        """
        Set up the test environment.

        Creates collection, ingests fixtures, builds pipeline.
        Uses tests/test_config.yaml (same as all other tests).

        Args:
            tier_name: Which tier to initialize with (default: first tier)

        Returns:
            Duration of ingestion in seconds
        """
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

        # Load test config (same as all other tests)
        e2e_config = load_e2e_config()
        tier_names = get_tier_names(e2e_config)
        tier_name = tier_name or tier_names[0]  # Default to first tier

        logger.info(f"E2E Setup: Using test_config.yaml (tier: {tier_name})")
        logger.info(f"E2E Setup: Available tiers: {tier_names}")

        # Get tier-specific config (returns nested structure for compatibility)
        tier_config = get_tier_config(tier_name, e2e_config)

        # Get plugin names from tier config
        chat_plugin = tier_config["chat"]["plugin_name"]
        chat_kwargs = tier_config["chat"].get("kwargs", {})
        embedding_plugin = tier_config["embedding"]["plugin_name"]
        embedding_kwargs = tier_config["embedding"].get("kwargs", {})
        vector_db_plugin = tier_config["vector_db"]["plugin_name"]
        vector_db_kwargs = tier_config["vector_db"].get("kwargs", {})

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
            # Core plugins (string format per new schema)
            "chat": chat_plugin,
            "embedding": embedding_plugin,
            "vector_db": vector_db_plugin,
            # Required field
            "collection": self.collection,
            # Retrieval settings (flat structure)
            "retrieval_plugin": "dense",
            "top_k": 40,  # Higher for better recall with local embeddings
            # Generation settings (flat, not nested under rgs)
            "strict_grounding": False,  # Allow more flexible answers in tests
            "max_chunks": 50,  # Higher limit to accommodate aggregation queries (3x multiplier)
            # Plugin kwargs
            "chat_kwargs": chat_kwargs,
            "embedding_kwargs": embedding_kwargs,
            "vector_db_kwargs": vector_db_kwargs,
        }

        cfg = FitzRagConfig(**config_dict)
        # Disable default constraints for E2E tests to isolate retrieval testing
        # (constraints=[] prevents InsufficientEvidence/ConflictAware from blocking)
        # Disable keywords - auto-detected vocabulary filters out valid results on small corpus
        self.pipeline = RAGPipeline.from_config(cfg, constraints=[], enable_keywords=False)

        self._setup_complete = True
        self._current_tier = tier_name
        logger.info(f"E2E Setup: Complete (tier: {tier_name})")

        return ingestion_duration

    def _rebuild_pipeline(self, tier_name: str) -> None:
        """
        Rebuild pipeline with a different tier's chat model.

        Only swaps the chat/LLM component - reuses existing vector client,
        embedder, and retrieval infrastructure to avoid losing the ingested collection.

        Args:
            tier_name: Name of the tier to switch to
        """
        from fitz_ai.llm.registry import get_llm_plugin

        if tier_name == self._current_tier:
            logger.debug(f"Already on tier '{tier_name}', skipping rebuild")
            return

        logger.info(f"E2E: Switching to tier '{tier_name}'")

        e2e_config = load_e2e_config()
        tier_config = get_tier_config(tier_name, e2e_config)

        # Get the new chat plugin for this tier
        chat_plugin_name = tier_config["chat"]["plugin_name"]
        chat_kwargs = tier_config["chat"].get("kwargs", {})

        new_chat = get_llm_plugin(
            plugin_type="chat",
            plugin_name=chat_plugin_name,
            **chat_kwargs,
        )

        # Also get fast chat for multi-hop (uses same tier)
        new_fast_chat = get_llm_plugin(
            plugin_type="chat",
            plugin_name=chat_plugin_name,
            tier="fast",
            **chat_kwargs,
        )

        # Swap chat models in existing pipeline (keeps vector client, retrieval, etc.)
        self.pipeline.chat = new_chat

        # Update hop controller if it exists (for multi-hop queries)
        if self.pipeline.hop_controller is not None:
            self.pipeline.hop_controller.evaluator.chat = new_fast_chat
            self.pipeline.hop_controller.extractor.chat = new_fast_chat

        self._current_tier = tier_name
        logger.info(f"E2E: Switched to tier '{tier_name}' (chat={chat_plugin_name})")

    def _cleanup_stale_data(self) -> None:
        """
        Clean up any stale data from previous test runs.

        Called at the start of setup() to ensure a clean slate, even if
        a previous test run was interrupted and teardown didn't execute.
        """
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        try:
            # Need vector client to delete collection - use test config
            config = load_e2e_config()
            vector_db_plugin = config["vector_db"]
            vector_db_kwargs = config.get("vector_db_kwargs", {})

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
                logger.debug("E2E Setup: Cleaned up stale table collection")
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

    def run_scenario(self, scenario: TestScenario, use_cache: bool = True) -> ScenarioResult:
        """
        Run a single test scenario.

        Args:
            scenario: The scenario to run
            use_cache: Whether to check/update cache (default: True)

        Returns:
            ScenarioResult with validation outcome
        """
        if not self._setup_complete:
            raise RuntimeError("E2E runner not set up. Call setup() first.")

        logger.debug(f"Running scenario {scenario.id}: {scenario.name}")

        start_time = time.time()

        # Check cache first (file-based, one .txt per scenario)
        # Cache hit only returns if the cached result passed - failed results aren't cached
        if use_cache and self.cache.enabled:
            cached = self.cache.get(scenario.query, [], scenario_id=scenario.id)
            if cached and cached.get("passed"):
                logger.debug(f"Cache hit for scenario {scenario.id} (tier={cached.get('tier')})")
                duration_ms = (time.time() - start_time) * 1000
                return ScenarioResult(
                    scenario=scenario,
                    validation=ValidationResult(
                        passed=True,
                        reason="Cached result",
                        details={"cached": True, "original_tier": cached.get("tier")},
                    ),
                    answer_text=cached.get("answer_text", "")[:500],
                    duration_ms=duration_ms,
                    error=None,
                )

        # Cache miss or disabled - run the pipeline
        error = None
        answer_text = ""
        chunk_ids: list[str] = []

        try:
            answer = self.pipeline.run(scenario.query)
            answer_text = answer.answer
            # Extract chunk IDs from sources/provenance
            sources = getattr(answer, "sources", None) or getattr(answer, "provenance", [])
            chunk_ids = [s.source_id for s in sources if hasattr(s, "source_id")]
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

        # Cache successful results only (failed results should be retried)
        if use_cache and self.cache.enabled and validation.passed:
            self.cache.set(
                query=scenario.query,
                chunk_ids=chunk_ids,
                scenario_id=scenario.id,
                answer_text=answer_text[:500] if answer_text else "",
                passed=True,
                tier=self._current_tier or "unknown",
            )

        return ScenarioResult(
            scenario=scenario,
            validation=validation,
            answer_text=answer_text[:500] if answer_text else "",
            duration_ms=duration_ms,
            error=error,
            chunk_ids=chunk_ids,
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

    def run_tiered(self, scenarios: list[TestScenario] | None = None) -> TieredRunResult:
        """
        Run tests with tiered fallback.

        Executes scenarios through tiers in order:
        1. Check cache for previously passed results
        2. Run remaining with first tier (fast/local)
        3. Re-run failures with second tier (cloud)
        4. Continue until all tiers exhausted or all passed

        Args:
            scenarios: Optional list of scenarios (default: all SCENARIOS)

        Returns:
            TieredRunResult with results and tier assignments
        """
        if not self._setup_complete:
            raise RuntimeError("E2E runner not set up. Call setup() first.")

        scenarios = scenarios or SCENARIOS
        e2e_config = load_e2e_config()
        tier_names = get_tier_names(e2e_config)

        all_results: dict[str, tuple[ScenarioResult, str]] = {}
        remaining = list(scenarios)

        logger.info(f"E2E Tiered Run: Starting {len(scenarios)} scenarios")
        logger.info(f"E2E Tiered Run: Tiers = {tier_names}")
        logger.info(f"E2E Tiered Run: Cache enabled = {self.cache.enabled}")
        start_time = time.time()

        # Phase 0: Check cache for previously passed results (file-based)
        if self.cache.enabled:
            logger.info("\n--- Checking cache ---")
            still_need_run = []
            for scenario in remaining:
                cached = self.cache.get(scenario.query, [], scenario_id=scenario.id)
                if cached and cached.get("passed"):
                    # Return cached result
                    result = ScenarioResult(
                        scenario=scenario,
                        validation=ValidationResult(
                            passed=True,
                            reason="Cached result",
                            details={"cached": True, "original_tier": cached.get("tier")},
                        ),
                        answer_text=cached.get("answer_text", "")[:500],
                        duration_ms=0.0,
                        error=None,
                    )
                    all_results[scenario.id] = (result, "cached")
                    logger.info(f"  [CACHE] {scenario.id}: {scenario.name}")
                else:
                    still_need_run.append(scenario)

            cached_count = len(remaining) - len(still_need_run)
            logger.info(f"Cache: {cached_count} hits, {len(still_need_run)} need to run")
            remaining = still_need_run

        # Track last result for scenarios that fail all tiers
        last_results: dict[str, ScenarioResult] = {}

        # Iterate through tiers
        for tier_name in tier_names:
            if not remaining:
                break

            logger.info(f"\n--- Tier '{tier_name}': {len(remaining)} scenarios ---")
            self._rebuild_pipeline(tier_name)

            still_failing = []
            for scenario in remaining:
                # Don't use cache in run_scenario - we already checked above
                result = self.run_scenario(scenario, use_cache=False)
                last_results[scenario.id] = result  # Track for failed_all

                if result.validation.passed:
                    all_results[scenario.id] = (result, tier_name)
                    status = "PASS"
                    # Cache successful result
                    if self.cache.enabled:
                        self.cache.set(
                            query=scenario.query,
                            chunk_ids=result.chunk_ids,
                            scenario_id=scenario.id,
                            answer_text=result.answer_text,
                            passed=True,
                            tier=tier_name,
                        )
                else:
                    still_failing.append(scenario)
                    status = "FAIL"

                logger.info(
                    f"  [{status}] {scenario.id}: {scenario.name} ({result.duration_ms:.0f}ms)"
                )

            passed_this_tier = len(remaining) - len(still_failing)
            logger.info(f"Tier '{tier_name}': {passed_this_tier}/{len(remaining)} passed")

            remaining = still_failing

        # Any remaining scenarios failed all tiers - use their last result
        for scenario in remaining:
            result = last_results[scenario.id]
            all_results[scenario.id] = (result, "failed_all")

        total_duration = time.time() - start_time

        tiered_result = TieredRunResult(
            results=all_results,
            tier_names=tier_names,
            cache_stats=self.cache.stats(),
            total_duration_s=total_duration,
        )

        tiered_result.print_summary()

        return tiered_result


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


def run_e2e_tests_tiered(
    fixtures_dir: Path | None = None,
    scenarios: list[TestScenario] | None = None,
) -> TieredRunResult:
    """
    Convenience function to run E2E tests with tiered fallback.

    Runs tests through multiple tiers (local -> cloud), cascading
    failures to higher tiers.

    Args:
        fixtures_dir: Path to fixtures (default: tests/e2e/fixtures)
        scenarios: Optional specific scenarios to run

    Returns:
        TieredRunResult with tier assignments
    """
    runner = E2ERunner(fixtures_dir=fixtures_dir)

    try:
        runner.setup()
        result = runner.run_tiered(scenarios)
        return result
    finally:
        runner.teardown()


if __name__ == "__main__":
    # Run E2E tests from command line
    import sys

    tiered_mode = "--tiered" in sys.argv

    print("=" * 60)
    print("E2E Retrieval Intelligence Tests")
    if tiered_mode:
        print("(TIERED MODE: local -> cloud fallback)")
    print("=" * 60)
    print()

    if tiered_mode:
        result = run_e2e_tests_tiered()
        # TieredRunResult.print_summary() already called
        sys.exit(0 if result.total_failed == 0 else 1)
    else:
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
