# tests/e2e/scenarios.py
"""
E2E test scenario definitions.

Each scenario tests a specific retrieval intelligence feature with a query
and expected outcome validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Feature(Enum):
    """Retrieval intelligence features under test."""

    MULTI_HOP = "multi_hop"
    ENTITY_GRAPH = "entity_graph"
    COMPARISON = "comparison"
    MULTI_QUERY = "multi_query"
    KEYWORD_EXACT = "keyword_exact"
    CONFLICT_AWARE = "conflict_aware"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CAUSAL_ATTRIBUTION = "causal_attribution"
    TABLE_SCHEMA = "table_schema"
    TABLE_QUERY = "table_query"
    CODE_SEARCH = "code_search"
    LONG_DOC = "long_doc"
    DEDUP = "dedup"
    BASIC_RETRIEVAL = "basic_retrieval"


@dataclass
class TestScenario:
    """
    A single E2E test scenario.

    Attributes:
        id: Unique scenario identifier (e.g., "E01")
        name: Human-readable scenario name
        feature: The retrieval feature being tested
        query: The query to run against the RAG pipeline
        must_contain: Substrings that MUST appear in the answer (case-insensitive)
        must_contain_any: At least ONE of these must appear (case-insensitive)
        must_not_contain: Substrings that must NOT appear in the answer
        expected_mode: Expected AnswerMode (e.g., "HEDGED", "ABSTAIN")
        min_sources: Minimum number of source citations expected
        custom_validator: Optional custom validation function
    """

    id: str
    name: str
    feature: Feature
    query: str
    must_contain: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    expected_mode: Optional[str] = None
    min_sources: int = 0
    custom_validator: Optional[Callable] = None


# =============================================================================
# Test Scenarios
# =============================================================================

SCENARIOS: list[TestScenario] = [
    # =========================================================================
    # Multi-Hop Retrieval
    # =========================================================================
    TestScenario(
        id="E01",
        name="Multi-hop: competitor product through person",
        feature=Feature.MULTI_HOP,
        query="What does Sarah Chen's company's main competitor manufacture?",
        must_contain_any=["hydrogen", "fuel cell"],
        min_sources=1,
    ),
    TestScenario(
        id="E02",
        name="Multi-hop: competitor CEO",
        feature=Feature.MULTI_HOP,
        query="Who is the CEO of TechCorp's main competitor?",
        must_contain=["Marcus Webb"],
        min_sources=1,
    ),
    # =========================================================================
    # Entity Graph Expansion
    # =========================================================================
    TestScenario(
        id="E03",
        name="Entity graph: company info includes related entities",
        feature=Feature.ENTITY_GRAPH,
        query="Tell me about TechCorp Industries",
        must_contain_any=["Sarah Chen", "electric vehicle", "GreenDrive"],
        min_sources=1,
    ),
    TestScenario(
        id="E04",
        name="Entity graph: person to company connection",
        feature=Feature.ENTITY_GRAPH,
        query="What company does Sarah Chen work for and what do they make?",
        must_contain=["TechCorp"],
        must_contain_any=["electric", "vehicle", "battery"],
        min_sources=1,
    ),
    # =========================================================================
    # Comparison Detection
    # =========================================================================
    TestScenario(
        id="E05",
        name="Comparison: two products",
        feature=Feature.COMPARISON,
        query="Compare the Model X100 vs Model Y200",
        must_contain=["X100", "Y200"],
        must_contain_any=["price", "range", "battery"],
        min_sources=1,
    ),
    TestScenario(
        id="E06",
        name="Comparison: price difference",
        feature=Feature.COMPARISON,
        query="What is the price difference between Model X100 and Model Z50?",
        must_contain_any=["45,000", "35,000", "10,000", "$10"],
        min_sources=1,
    ),
    # =========================================================================
    # Multi-Query Expansion
    # =========================================================================
    TestScenario(
        id="E07",
        name="Multi-query: complex question about multiple attributes",
        feature=Feature.MULTI_QUERY,
        query="What are the battery capacities and ranges for all TechCorp vehicle models?",
        must_contain_any=["75 kWh", "100 kWh", "50 kWh"],
        min_sources=1,
    ),
    # =========================================================================
    # Conflict Detection
    # =========================================================================
    TestScenario(
        id="E08",
        name="Conflict: employee count discrepancy",
        feature=Feature.CONFLICT_AWARE,
        query="How many employees does the company have?",
        # Should mention the discrepancy between different reports
        must_contain_any=["5,200", "4,800", "5,500", "differ", "conflict", "discrepancy", "varies"],
        min_sources=1,
    ),
    TestScenario(
        id="E09",
        name="Conflict: revenue figures",
        feature=Feature.CONFLICT_AWARE,
        query="What was the Q1 2024 revenue?",
        must_contain_any=["1.2 billion", "1.4 billion", "differ", "discrepancy"],
        min_sources=1,
    ),
    # =========================================================================
    # Insufficient Evidence
    # =========================================================================
    TestScenario(
        id="E10",
        name="Insufficient: missing budget information",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is Project Alpha's budget?",
        # Should indicate information is not available
        must_contain_any=[
            "don't know",
            "not available",
            "not provided",
            "not found",
            "no information",
            "don't have",
            "classified",
            "restricted",
            "confidential",
        ],
        min_sources=0,
    ),
    TestScenario(
        id="E11",
        name="Insufficient: missing timeline",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="When will Project Alpha be completed?",
        must_contain_any=[
            "don't know",
            "not available",
            "not provided",
            "not found",
            "no information",
            "don't have",
            "classified",
            "restricted",
            "do not contain",
            "cannot answer",
            "unable to",
        ],
        min_sources=0,
    ),
    # =========================================================================
    # Causal Attribution
    # =========================================================================
    TestScenario(
        id="E12",
        name="Causal: stock price rise",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Why did TechCorp's stock price rise 20% in Q1 2024?",
        # Should mention multiple factors, not claim single cause
        must_contain_any=[
            "product launch",
            "competitor",
            "tax credit",
            "analyst",
            "multiple",
            "factors",
            "several",
        ],
        min_sources=1,
    ),
    # =========================================================================
    # Table Schema / CSV Queries
    # =========================================================================
    TestScenario(
        id="E13",
        name="Table: column discovery",
        feature=Feature.TABLE_SCHEMA,
        query="What columns are in the employee data?",
        must_contain_any=["employee_id", "name", "department", "salary"],
        min_sources=1,
    ),
    TestScenario(
        id="E14",
        name="Table: department query",
        feature=Feature.TABLE_QUERY,
        query="How many employees are in the Engineering department?",
        must_contain_any=["5", "five", "Engineering"],
        min_sources=1,
    ),
    TestScenario(
        id="E15",
        name="Table: salary query",
        feature=Feature.TABLE_QUERY,
        query="What is the average salary in the Engineering department?",
        # Average of 95000, 105000, 98000, 110000, 88000 = 99200
        must_contain_any=["99", "100", "average"],
        min_sources=1,
    ),
    # =========================================================================
    # Code Search
    # =========================================================================
    TestScenario(
        id="E16",
        name="Code: class description",
        feature=Feature.CODE_SEARCH,
        query="What does the UserAuth class do?",
        must_contain_any=["authentication", "session", "login"],
        min_sources=1,
    ),
    TestScenario(
        id="E17",
        name="Code: method lookup",
        feature=Feature.CODE_SEARCH,
        query="How does the login method work in UserAuth?",
        must_contain_any=["authenticate", "password", "token", "session"],
        min_sources=1,
    ),
    # =========================================================================
    # Long Document Retrieval
    # =========================================================================
    TestScenario(
        id="E18",
        name="Long doc: authentication system",
        feature=Feature.LONG_DOC,
        query="How does the TechCorp authentication system work?",
        must_contain_any=["JWT", "token", "24-hour", "OAuth"],
        min_sources=1,
    ),
    TestScenario(
        id="E19",
        name="Long doc: database architecture",
        feature=Feature.LONG_DOC,
        query="How does the Data Service handle database connections?",
        must_contain_any=["PostgreSQL", "connection pool", "50 connections", "replica"],
        min_sources=1,
    ),
    # =========================================================================
    # Basic Retrieval (sanity checks)
    # =========================================================================
    TestScenario(
        id="E20",
        name="Basic: simple fact lookup",
        feature=Feature.BASIC_RETRIEVAL,
        query="Where is TechCorp Industries headquartered?",
        must_contain=["Austin"],
        min_sources=1,
    ),
    TestScenario(
        id="E21",
        name="Basic: product price",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the price of the Model Y200?",
        must_contain_any=["55,000", "$55"],
        min_sources=1,
    ),
]


def get_scenarios_by_feature(feature: Feature) -> list[TestScenario]:
    """Get all scenarios for a specific feature."""
    return [s for s in SCENARIOS if s.feature == feature]


def get_scenario_by_id(scenario_id: str) -> TestScenario | None:
    """Get a scenario by its ID."""
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None
