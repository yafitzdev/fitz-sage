# fitz_ai/evaluation/benchmarks/fitz_gov.py
"""
fitz-gov benchmark integration for Fitz RAG engine.

This module wraps the fitz-gov package to evaluate Fitz's governance calibration.
The evaluation logic lives in fitz-gov so all RAG systems get identical evaluation.

Requires: pip install fitz-gov (or pip install -e path/to/fitz-gov)

Usage:
    from fitz_ai.evaluation.benchmarks import FitzGovBenchmark

    benchmark = FitzGovBenchmark()
    results = benchmark.evaluate(engine)
    print(results)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_ai.engines.fitz_krag.engine import FitzKragEngine

logger = get_logger(__name__)

# Default data directory (downloaded from GitHub)
DATA_DIR = Path.home() / ".fitz" / "fitz_gov_data"


def _import_fitz_gov():
    """Lazy import fitz-gov package with helpful error message."""
    try:
        import fitz_gov

        return fitz_gov
    except ImportError as e:
        raise ImportError(
            "fitz-gov package is required for fitz-gov benchmark. "
            "Install with: pip install fitz-gov "
            "or: pip install -e path/to/fitz-gov"
        ) from e


# Lazy re-exports for backward compatibility
def __getattr__(name: str):
    """Lazy import types from fitz-gov."""
    if name in (
        "FitzGovCategory",
        "FitzGovCase",
        "FitzGovCaseResult",
        "FitzGovCategoryResult",
        "FitzGovConfusionMatrix",
        "FitzGovResult",
        "OllamaValidator",
        "ValidatorConfig",
        "ValidationResult",
    ):
        fitz_gov = _import_fitz_gov()
        return getattr(fitz_gov, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class FitzGovBenchmark:
    """
    fitz-gov governance calibration benchmark for Fitz RAG engine.

    Tests Fitz's ability to correctly classify when to:
    - Abstain (insufficient evidence)
    - Dispute (conflicting sources)
    - Answer trustworthily (clear evidence)

    Example:
        benchmark = FitzGovBenchmark()

        # Run full benchmark (downloads data from GitHub if needed)
        results = benchmark.evaluate(engine)

        # Run specific category
        from fitz_gov import FitzGovCategory
        results = benchmark.evaluate(engine, categories=[FitzGovCategory.ABSTENTION])
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        full_mode: bool = False,
        llm_validation: bool = False,
        llm_model: str = "qwen2.5:14b",
        llm_base_url: str = "http://localhost:11434",
        enrich_chunks: bool = True,
        use_fusion: bool = False,
        adaptive: bool = True,
        model_override: str | None = None,
    ):
        """
        Initialize fitz-gov benchmark.

        Args:
            data_dir: Directory containing test case JSON files.
                     Defaults to fitz-gov package data directory.
            full_mode: If True, run full LLM generation for answer quality tests.
                      If False (default), test governance mode classification only.
            llm_validation: If True, use two-pass validation (regex + LLM) for
                           grounding and relevance categories. Requires Ollama.
            llm_model: Ollama model for LLM validation. Default: qwen2.5:14b
            llm_base_url: Ollama API URL. Default: http://localhost:11434
            enrich_chunks: If True, enrich chunks with metadata (summary, keywords,
                          entities) before running constraints. This simulates the
                          full ingestion pipeline.
            use_fusion: If True, use 3-prompt fusion for contradiction detection.
                       Reduces variance via majority voting.
            adaptive: If True, auto-select detection method based on query type:
                     - Uncertainty/causal queries → Fusion (conservative)
                     - Factual queries → Standard pairwise (aggressive)
            model_override: Override chat model for constraints. Format: provider or
                          provider/model (e.g., "ollama", "ollama/qwen2.5:3b", "cohere").
                          If None, uses the engine's configured model.
        """
        # Import fitz-gov (validates it's installed)
        fitz_gov = _import_fitz_gov()

        self._data_dir = Path(data_dir) if data_dir else None
        self._full_mode = full_mode
        self._llm_validation = llm_validation
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._enrich_chunks = enrich_chunks
        self._use_fusion = use_fusion
        self._adaptive = adaptive
        self._model_override = model_override
        self._enricher = None
        self._chat_factory_override = None

        # Create chat factory override if model specified
        if model_override:
            from fitz_ai.llm.factory import get_chat_factory

            self._chat_factory_override = get_chat_factory(model_override)

        # Store fitz-gov module reference for lazy access
        self._fitz_gov = fitz_gov

        # Create evaluator from fitz-gov
        self._evaluator = fitz_gov.FitzGovEvaluator(
            llm_validation=llm_validation,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
        )

        # fitz-gov uses the same 3-mode system (TRUSTWORTHY/DISPUTED/ABSTAIN)
        self._mode_to_fitz_gov = {
            AnswerMode.ABSTAIN: fitz_gov.AnswerMode.ABSTAIN,
            AnswerMode.DISPUTED: fitz_gov.AnswerMode.DISPUTED,
            AnswerMode.TRUSTWORTHY: fitz_gov.AnswerMode.TRUSTWORTHY,
        }

        # Initialize enricher if enabled
        if enrich_chunks:
            self._init_enricher()

    def _init_enricher(self) -> None:
        """Initialize the chunk enricher for metadata extraction."""
        from fitz_ai.ingestion.enrichment.bus import ChunkEnricher
        from fitz_ai.ingestion.enrichment.modules import (
            EntityModule,
            KeywordModule,
            SummaryModule,
        )
        from fitz_ai.llm import get_chat_factory

        try:
            # Use ollama for enrichment (runs locally)
            chat_factory = get_chat_factory("ollama")
            self._enricher = ChunkEnricher(
                chat_factory=chat_factory,
                modules=[SummaryModule(), KeywordModule(), EntityModule()],
                min_batch_content=50,  # Lower threshold for benchmark's short chunks
            )
            logger.info("Initialized chunk enricher for fitz-gov benchmark")
        except Exception as e:
            logger.warning(f"Failed to initialize enricher: {e}")
            self._enricher = None

    def evaluate(
        self,
        engine: FitzKragEngine,
        categories: list | None = None,
        test_cases: list | None = None,
    ) -> Any:
        """
        Evaluate governance calibration.

        Args:
            engine: Fitz RAG engine to evaluate
            categories: Categories to test (FitzGovCategory). Defaults to all.
            test_cases: Custom test cases (FitzGovCase). If not provided, loads from data_dir.

        Returns:
            FitzGovResult with accuracy and confusion matrix
        """
        import time

        start_time = time.time()

        # Load test cases from fitz-gov package
        if test_cases is None:
            test_cases = self._fitz_gov.load_cases(categories, self._data_dir)

        # Filter by category if specified
        if categories:
            test_cases = [c for c in test_cases if c.category in categories]

        # Run engine on each case and collect responses + modes
        responses: list[str] = []
        modes: list = []

        for case in test_cases:
            answer_text, actual_mode = self._run_case(engine, case)
            responses.append(answer_text)
            modes.append(self._mode_to_fitz_gov.get(actual_mode) if actual_mode else None)

        # Use fitz-gov evaluator for evaluation
        result = self._evaluator.evaluate_all(test_cases, responses, modes)

        # Update timing
        result.evaluation_time_seconds = time.time() - start_time
        result.metadata["full_mode"] = self._full_mode
        result.metadata["use_fusion"] = self._use_fusion
        result.metadata["adaptive"] = self._adaptive
        if self._model_override:
            result.metadata["model"] = self._model_override

        return result

    def _run_case(self, engine: FitzKragEngine, case: Any) -> tuple[str, AnswerMode | None]:
        """
        Run engine on a single test case.

        Returns:
            Tuple of (answer_text, actual_mode)
        """
        from fitz_ai.core import Query

        query = Query(text=case.query)

        # Run with injected contexts (bypass retrieval for controlled testing)
        answer = self._run_with_contexts(engine, query, case.contexts)

        # Extract governance info
        actual_mode = self._get_answer_mode(answer)
        answer_text = answer.answer if hasattr(answer, "answer") else str(answer)

        return answer_text, actual_mode

    def _run_with_contexts(self, engine: FitzKragEngine, query, contexts: list[str]):
        """
        Run governance classification on injected contexts.

        In governance-only mode: tests constraint -> mode classification only.
        In full mode: runs LLM generation for answer quality evaluation.
        """
        from dataclasses import dataclass

        from fitz_ai.core import Chunk
        from fitz_ai.governance import AnswerGovernor
        from fitz_ai.governance.constraints.runner import run_constraints

        # Create chunks from injected contexts
        chunks = [
            Chunk(
                id=f"fitz_gov_context_{i}",
                doc_id="fitz_gov_test_doc",
                content=ctx,
                chunk_index=i,
                metadata={"source": "fitz_gov_benchmark"},
            )
            for i, ctx in enumerate(contexts)
        ]

        # Enrich chunks if enricher is available
        if self._enricher is not None:
            try:
                result = self._enricher.enrich(chunks)
                chunks = result.chunks
                logger.debug(f"Enriched {len(chunks)} chunks")
            except Exception as e:
                logger.warning(f"Chunk enrichment failed: {e}")

        # Step 1: Run constraints on injected chunks
        if self._use_fusion or self._adaptive or self._model_override:
            # Create constraints with fusion/adaptive mode for contradiction detection
            from fitz_ai.governance import (
                AnswerVerificationConstraint,
                CausalAttributionConstraint,
                ConflictAwareConstraint,
                InsufficientEvidenceConstraint,
                SpecificInfoTypeConstraint,
            )

            # Use model override if specified, otherwise use engine's chat factory
            if self._chat_factory_override:
                fast_chat = self._chat_factory_override("fast")
                balanced_chat = self._chat_factory_override("balanced")
            else:
                fast_chat = engine._chat_factory("fast")
                balanced_chat = engine._chat_factory("balanced")
            # Get embedder from engine for semantic relevance checking
            embedder = engine._embedder
            constraints = [
                InsufficientEvidenceConstraint(embedder=embedder, chat=fast_chat),
                SpecificInfoTypeConstraint(),
                CausalAttributionConstraint(),
                ConflictAwareConstraint(
                    chat=fast_chat,
                    use_fusion=self._use_fusion,
                    adaptive=self._adaptive,
                    embedder=embedder,
                ),
                AnswerVerificationConstraint(chat=fast_chat),
            ]
            constraint_results = run_constraints(query.text, chunks, constraints)
        else:
            constraint_results = run_constraints(query.text, chunks, engine._constraints)

        # Step 2: Get governance decision
        governor = AnswerGovernor()
        governance = governor.decide(constraint_results)

        # In governance-only mode, return minimal result (no LLM)
        if not self._full_mode:

            @dataclass
            class GovernanceResult:
                answer: str
                mode: AnswerMode
                triggered_constraints: set

            return GovernanceResult(
                answer=f"[Governance test - mode: {governance.mode.value}]",
                mode=governance.mode,
                triggered_constraints=governance.triggered_constraints,
            )

        # Full mode: run LLM generation for answer quality evaluation
        # Wrap chunks as ReadResult objects (KRAG's assembler/synthesizer expect ReadResult)
        from fitz_ai.engines.fitz_krag.types import Address, AddressKind, ReadResult

        read_results = [
            ReadResult(
                address=Address(
                    kind=AddressKind.SECTION,
                    source_id=chunk.id,
                    location="fitz_gov_benchmark",
                    summary=chunk.content[:100],
                ),
                content=chunk.content,
                file_path="fitz_gov_benchmark",
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]

        context = engine._assembler.assemble(query.text, read_results)
        answer = engine._synthesizer.generate(
            query.text, context, read_results, answer_mode=governance.mode
        )

        return answer

    def _get_answer_mode(self, answer) -> AnswerMode:
        """Extract answer mode from answer object."""
        if hasattr(answer, "mode"):
            return answer.mode
        if hasattr(answer, "governance") and hasattr(answer.governance, "mode"):
            return answer.governance.mode
        return AnswerMode.TRUSTWORTHY

    def get_available_categories(self) -> list[str]:
        """Get list of available categories."""
        return [c.value for c in self._fitz_gov.FitzGovCategory]
