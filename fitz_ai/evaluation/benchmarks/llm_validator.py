# fitz_ai/evaluation/benchmarks/llm_validator.py
"""
Two-pass LLM validation for FITZ-GOV benchmark.

Pass 1: Regex catches obvious violations (fast)
Pass 2: LLM validates flagged cases to reduce false positives (semantic)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of LLM validation."""

    is_violation: bool
    confidence: float  # 0.0-1.0
    reasoning: str
    cached: bool = False
    latency_ms: float = 0.0


@dataclass
class ValidatorConfig:
    """Configuration for LLM validator."""

    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    timeout: float = 30.0
    temperature: float = 0.0  # Deterministic
    cache_enabled: bool = True
    cache_dir: Path | None = None
    cache_ttl_days: int = 7
    on_error: Literal["fail_open", "fail_closed"] = "fail_open"


# Prompt templates
FORBIDDEN_CLAIM_PROMPT = """You are evaluating whether a RAG system hallucinated information.

CONTEXT PROVIDED TO SYSTEM:
{context}

QUESTION ASKED:
{query}

SYSTEM'S RESPONSE:
{response}

FLAGGED TEXT: "{matched_text}"
PATTERN MATCHED: {pattern}
WHY FLAGGED: {rationale}

The regex flagged "{matched_text}" as a potential hallucination because it matches a forbidden pattern.

Determine if this is:
1. TRUE_VIOLATION - The system invented/hallucinated specific information that is NOT in the context
2. FALSE_POSITIVE - The flagged text is legitimate because:
   - It's stating what is NOT known/specified (e.g., "no price mentioned")
   - It's quoting or paraphrasing the actual context
   - It's discussing the absence of information
   - The pattern matched incidentally in valid text

Think step by step:
1. Is the flagged text claiming specific information?
2. Is that information actually in the context?
3. Or is the response acknowledging what's NOT available?

Respond ONLY with valid JSON (no markdown):
{{"verdict": "TRUE_VIOLATION" or "FALSE_POSITIVE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


REQUIRED_ELEMENT_PROMPT = """You are evaluating whether a RAG system properly acknowledged missing information.

CONTEXT PROVIDED TO SYSTEM:
{context}

QUESTION ASKED:
{query}

SYSTEM'S RESPONSE:
{response}

EXPECTED: The response should indicate that the specific information requested is not available in the context.
LOOKING FOR: Language like "{required_element}" or equivalent acknowledgment.

Determine if the response:
1. ACKNOWLEDGES - Properly notes the requested information is missing/not specified/not available
2. FAILS_TO_ACKNOWLEDGE - Either:
   - Does not mention that information is missing
   - Wrongly claims to have the answer
   - Only discusses related information without addressing the gap

Think step by step:
1. What specific information was asked for?
2. Is that information in the context?
3. Does the response acknowledge what's missing?

Respond ONLY with valid JSON (no markdown):
{{"verdict": "ACKNOWLEDGES" or "FAILS_TO_ACKNOWLEDGE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


FORBIDDEN_ELEMENT_PROMPT = """You are evaluating whether a RAG system incorrectly claimed to have information it doesn't have.

CONTEXT PROVIDED TO SYSTEM:
{context}

QUESTION ASKED:
{query}

SYSTEM'S RESPONSE:
{response}

FLAGGED TEXT: "{matched_text}"
WHY FLAGGED: This pattern suggests the system is claiming to provide specific information that isn't in the context.

Determine if this is:
1. TRUE_VIOLATION - The system is falsely claiming to have/provide specific information not in context
2. FALSE_POSITIVE - The flagged text is legitimate because:
   - It's actually in the context
   - It's discussing what the context DOES contain (vs what was asked)
   - The pattern matched incidentally

Respond ONLY with valid JSON (no markdown):
{{"verdict": "TRUE_VIOLATION" or "FALSE_POSITIVE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""


class OllamaValidator:
    """LLM-based validator using Ollama for semantic verification."""

    def __init__(self, config: ValidatorConfig | None = None):
        self.config = config or ValidatorConfig()
        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )
        default_cache = Path.home() / ".fitz" / "cache" / "llm_validation"
        self._cache_dir = self.config.cache_dir or default_cache
        if self.config.cache_enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        if self._available is not None:
            return self._available

        try:
            response = self._client.get("/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                model_base = self.config.model.split(":")[0]
                self._available = model_base in model_names or any(
                    self.config.model in m.get("name", "") for m in models
                )
                if not self._available:
                    logger.warning(
                        f"Model {self.config.model} not found in Ollama. "
                        f"Available: {model_names}. Run: ollama pull {self.config.model}"
                    )
            else:
                self._available = False
        except httpx.ConnectError:
            logger.warning(f"Ollama not running at {self.config.base_url}")
            self._available = False
        except Exception as e:
            logger.warning(f"Error checking Ollama availability: {e}")
            self._available = False

        return self._available

    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _get_cached(self, cache_key: str) -> ValidationResult | None:
        """Get cached result if available and not expired."""
        if not self.config.cache_enabled:
            return None

        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        try:
            data = json.loads(cache_file.read_text())
            # Check TTL
            cached_time = data.get("timestamp", 0)
            ttl_seconds = self.config.cache_ttl_days * 24 * 60 * 60
            if time.time() - cached_time > ttl_seconds:
                cache_file.unlink()
                return None

            return ValidationResult(
                is_violation=data["is_violation"],
                confidence=data["confidence"],
                reasoning=data["reasoning"],
                cached=True,
                latency_ms=0.0,
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def _set_cached(self, cache_key: str, result: ValidationResult) -> None:
        """Cache a validation result."""
        if not self.config.cache_enabled:
            return

        cache_file = self._cache_dir / f"{cache_key}.json"
        data = {
            "is_violation": result.is_violation,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "timestamp": time.time(),
        }
        cache_file.write_text(json.dumps(data, indent=2))

    def _call_ollama(self, prompt: str) -> dict:
        """Call Ollama API and parse JSON response."""
        response = self._client.post(
            "/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                },
            },
        )
        response.raise_for_status()

        result_text = response.json().get("response", "")

        # Parse JSON from response (handle potential markdown wrapping)
        result_text = result_text.strip()
        if result_text.startswith("```"):
            # Remove markdown code blocks
            lines = result_text.split("\n")
            result_text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        return json.loads(result_text)

    def validate_forbidden_claim(
        self,
        response: str,
        matched_text: str,
        pattern: str,
        context: str,
        query: str,
        rationale: str,
    ) -> ValidationResult:
        """
        Validate whether a regex-flagged forbidden claim is a true violation.

        Args:
            response: The LLM's response being evaluated
            matched_text: The text that matched the forbidden pattern
            pattern: The regex pattern that matched
            context: The context provided to the LLM
            query: The user's question
            rationale: Why this pattern is forbidden

        Returns:
            ValidationResult indicating if this is a true violation or false positive
        """
        prompt = FORBIDDEN_CLAIM_PROMPT.format(
            context=context,
            query=query,
            response=response,
            matched_text=matched_text,
            pattern=pattern,
            rationale=rationale,
        )

        return self._validate(prompt, violation_verdict="TRUE_VIOLATION")

    def validate_required_element(
        self,
        response: str,
        required_element: str,
        context: str,
        query: str,
    ) -> ValidationResult:
        """
        Validate whether response properly acknowledges missing information.

        Args:
            response: The LLM's response being evaluated
            required_element: The acknowledgment phrase expected
            context: The context provided to the LLM
            query: The user's question

        Returns:
            ValidationResult indicating if response acknowledges the gap
        """
        prompt = REQUIRED_ELEMENT_PROMPT.format(
            context=context,
            query=query,
            response=response,
            required_element=required_element,
        )

        return self._validate(prompt, violation_verdict="FAILS_TO_ACKNOWLEDGE")

    def validate_forbidden_element(
        self,
        response: str,
        matched_text: str,
        pattern: str,
        context: str,
        query: str,
    ) -> ValidationResult:
        """
        Validate whether a forbidden element match is a true violation.

        Args:
            response: The LLM's response being evaluated
            matched_text: The text that matched the forbidden pattern
            pattern: The regex pattern that matched
            context: The context provided to the LLM
            query: The user's question

        Returns:
            ValidationResult indicating if this is a true violation or false positive
        """
        prompt = FORBIDDEN_ELEMENT_PROMPT.format(
            context=context,
            query=query,
            response=response,
            matched_text=matched_text,
        )

        return self._validate(prompt, violation_verdict="TRUE_VIOLATION")

    def _validate(self, prompt: str, violation_verdict: str) -> ValidationResult:
        """Common validation logic."""
        # Check cache first
        cache_key = self._cache_key(prompt)
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Cache hit for validation: {cache_key}")
            return cached

        # Check availability
        if not self.is_available():
            logger.warning("Ollama not available, using fallback behavior")
            return self._fallback_result()

        # Call LLM
        start_time = time.time()
        try:
            result_data = self._call_ollama(prompt)
            latency_ms = (time.time() - start_time) * 1000

            verdict = result_data.get("verdict", "")
            is_violation = verdict == violation_verdict

            result = ValidationResult(
                is_violation=is_violation,
                confidence=float(result_data.get("confidence", 0.5)),
                reasoning=result_data.get("reasoning", ""),
                cached=False,
                latency_ms=latency_ms,
            )

            # Cache the result
            self._set_cached(cache_key, result)

            logger.debug(
                f"LLM validation: {verdict} (confidence={result.confidence:.2f}, "
                f"latency={latency_ms:.0f}ms)"
            )
            return result

        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout after {self.config.timeout}s")
            return self._fallback_result()
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return self._fallback_result()
        except Exception as e:
            logger.warning(f"LLM validation error: {e}")
            return self._fallback_result()

    def _fallback_result(self) -> ValidationResult:
        """Return fallback result based on config."""
        if self.config.on_error == "fail_open":
            # Trust regex (assume it's correct)
            return ValidationResult(
                is_violation=True,
                confidence=0.5,
                reasoning="LLM unavailable, trusting regex result",
                cached=False,
            )
        else:
            # fail_closed - be lenient
            return ValidationResult(
                is_violation=False,
                confidence=0.5,
                reasoning="LLM unavailable, assuming false positive",
                cached=False,
            )

    def clear_cache(self) -> int:
        """Clear all cached validation results. Returns number of files deleted."""
        if not self._cache_dir.exists():
            return 0

        count = 0
        for cache_file in self._cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached validation results")
        return count

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> OllamaValidator:
        return self

    def __exit__(self, *args) -> None:
        self.close()
