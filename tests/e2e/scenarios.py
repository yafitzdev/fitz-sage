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
    FRESHNESS = "freshness"
    HYBRID_SEARCH = "hybrid_search"
    QUERY_EXPANSION = "query_expansion"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"


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
    # =========================================================================
    # Keyword Exact Match
    # =========================================================================
    TestScenario(
        id="E22",
        name="Keyword: unique company name",
        feature=Feature.KEYWORD_EXACT,
        query="What is GreenDrive Inc's annual revenue?",
        must_contain_any=["2.3 billion", "$2.3"],
        min_sources=1,
    ),
    TestScenario(
        id="E23",
        name="Keyword: specific product model",
        feature=Feature.KEYWORD_EXACT,
        query="What is the 0-60 time for Model Z50?",
        must_contain_any=["6.8", "seconds"],
        min_sources=1,
    ),
    TestScenario(
        id="E24",
        name="Keyword: unique person name",
        feature=Feature.KEYWORD_EXACT,
        query="Where did Marcus Webb work before GreenDrive?",
        must_contain_any=["Toyota", "fuel cell"],
        min_sources=1,
    ),
    # =========================================================================
    # Deduplication
    # =========================================================================
    TestScenario(
        id="E25",
        name="Dedup: term appearing in multiple docs",
        feature=Feature.DEDUP,
        query="What products does TechCorp make?",
        # TechCorp mentioned in people.md, products.md, conflicts.md - should dedupe
        must_contain_any=["electric vehicle", "battery", "Model"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Table Queries
    # =========================================================================
    TestScenario(
        id="E26",
        name="Table: location filter",
        feature=Feature.TABLE_QUERY,
        query="Which employees work in the Nevada location?",
        must_contain_any=["Eva Brown", "Nathan Park", "Nevada"],
        min_sources=1,
    ),
    TestScenario(
        id="E27",
        name="Table: highest salary",
        feature=Feature.TABLE_QUERY,
        query="Who has the highest salary in the employee data?",
        # James Wilson at $145,000 or Peter Adams at $140,000
        must_contain_any=["James Wilson", "145,000", "Peter Adams", "140,000"],
        min_sources=1,
    ),
    TestScenario(
        id="E28",
        name="Table: department count",
        feature=Feature.TABLE_QUERY,
        query="How many departments are there in the employee data?",
        # Engineering, Marketing, Sales, HR, Finance = 5
        must_contain_any=["5", "five", "Engineering", "Marketing", "Sales", "HR", "Finance"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Code Search
    # =========================================================================
    TestScenario(
        id="E29",
        name="Code: exception classes",
        feature=Feature.CODE_SEARCH,
        query="What exceptions are defined in the authentication module?",
        must_contain_any=["AuthenticationError", "SessionExpiredError"],
        min_sources=1,
    ),
    TestScenario(
        id="E30",
        name="Code: RoleAuthorizer permissions",
        feature=Feature.CODE_SEARCH,
        query="What permissions does an admin role have?",
        must_contain_any=["read", "write", "delete", "manage_users"],
        min_sources=1,
    ),
    TestScenario(
        id="E31",
        name="Code: session expiration",
        feature=Feature.CODE_SEARCH,
        query="How long do sessions last in UserAuth by default?",
        must_contain_any=["24", "hour", "session_duration"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Long Document
    # =========================================================================
    TestScenario(
        id="E32",
        name="Long doc: vehicle telemetry",
        feature=Feature.LONG_DOC,
        query="How does TechCorp process vehicle telemetry data?",
        must_contain_any=["Kafka", "Flink", "TimescaleDB", "500,000"],
        min_sources=1,
    ),
    TestScenario(
        id="E33",
        name="Long doc: payment security",
        feature=Feature.LONG_DOC,
        query="What security measures does the Payment Service use?",
        must_contain_any=["PCI DSS", "tokenized", "Stripe", "3D Secure"],
        min_sources=1,
    ),
    TestScenario(
        id="E34",
        name="Long doc: notification channels",
        feature=Feature.LONG_DOC,
        query="What notification channels does TechCorp support?",
        must_contain_any=["push", "email", "SMS", "Firebase", "SendGrid", "Twilio"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Multi-Hop
    # =========================================================================
    TestScenario(
        id="E35",
        name="Multi-hop: person's previous employer product",
        feature=Feature.MULTI_HOP,
        query="What kind of vehicles does Sarah Chen's previous employer make?",
        # Sarah Chen → AutoMotors → gasoline-powered vehicles
        must_contain_any=["gasoline", "traditional", "AutoMotors"],
        min_sources=1,
    ),
    TestScenario(
        id="E36",
        name="Multi-hop: competitor headquarters",
        feature=Feature.MULTI_HOP,
        query="Where is TechCorp's main competitor headquartered?",
        # TechCorp → GreenDrive → San Jose
        must_contain=["San Jose"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Comparison
    # =========================================================================
    TestScenario(
        id="E37",
        name="Comparison: three models",
        feature=Feature.COMPARISON,
        query="Compare the range of all three TechCorp models",
        must_contain_any=["300", "400", "200"],  # miles for X100, Y200, Z50
        min_sources=1,
    ),
    TestScenario(
        id="E38",
        name="Comparison: feature difference",
        feature=Feature.COMPARISON,
        query="What features does Model Y200 have that Model Z50 doesn't?",
        must_contain_any=["self-driving", "solar", "Bang & Olufsen", "heads-up", "17-inch"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Causal Attribution
    # =========================================================================
    TestScenario(
        id="E39",
        name="Causal: competitor stock decline",
        feature=Feature.CAUSAL_ATTRIBUTION,
        query="Why did GreenDrive's stock decline in Q1 2024?",
        must_contain_any=["recall", "50,000", "cost-cutting", "confidence"],
        min_sources=1,
    ),
    # =========================================================================
    # Additional Basic Retrieval
    # =========================================================================
    TestScenario(
        id="E40",
        name="Basic: founding year",
        feature=Feature.BASIC_RETRIEVAL,
        query="When was TechCorp Industries founded?",
        must_contain=["2015"],
        min_sources=1,
    ),
    TestScenario(
        id="E41",
        name="Basic: battery warranty",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the warranty on TechCorp batteries?",
        must_contain_any=["8-year", "150,000 mile", "8 year"],
        min_sources=1,
    ),
    TestScenario(
        id="E42",
        name="Basic: Project Alpha team size",
        feature=Feature.BASIC_RETRIEVAL,
        query="How many engineers are on the Project Alpha team?",
        must_contain=["5"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Query Robustness
    # =========================================================================
    TestScenario(
        id="E43",
        name="Edge: case insensitivity",
        feature=Feature.BASIC_RETRIEVAL,
        query="WHERE IS TECHCORP INDUSTRIES HEADQUARTERED?",
        must_contain=["Austin"],
        min_sources=1,
    ),
    TestScenario(
        id="E44",
        name="Edge: typo in query",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the price of the Model Y2OO?",  # O instead of 0
        must_contain_any=["55,000", "$55", "Y200"],
        min_sources=1,
    ),
    TestScenario(
        id="E45",
        name="Edge: very short query",
        feature=Feature.BASIC_RETRIEVAL,
        query="TechCorp CEO",
        must_contain=["Sarah Chen"],
        min_sources=1,
    ),
    TestScenario(
        id="E46",
        name="Edge: question without question mark",
        feature=Feature.BASIC_RETRIEVAL,
        query="Tell me the headquarters location of TechCorp",
        must_contain=["Austin"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Negation and Exclusion
    # =========================================================================
    TestScenario(
        id="E47",
        name="Edge: negation query",
        feature=Feature.COMPARISON,
        query="Which TechCorp model does NOT have a solar roof option?",
        # Only Y200 has solar roof, so X100 and Z50 don't
        must_contain_any=["X100", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Edge Cases: Out of Domain
    # =========================================================================
    TestScenario(
        id="E48",
        name="Edge: completely out of domain",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is the recipe for chocolate cake?",
        must_contain_any=[
            "don't know",
            "no information",
            "not found",
            "cannot",
            "don't have",
            "not available",
            "outside",
            "beyond",
            "unrelated",
            "no recipe",
            "not contain",
            "unable",
            "doesn't",
            "does not",
            "no mention",
            "no data",
        ],
        min_sources=0,
    ),
    TestScenario(
        id="E49",
        name="Edge: plausible but missing info",
        feature=Feature.INSUFFICIENT_EVIDENCE,
        query="What is TechCorp's stock ticker symbol?",
        # TECH is mentioned in causal_claims.md but only as hypothetical
        must_contain_any=[
            "TECH",
            "don't know",
            "not provided",
            "not specified",
            "not found",
        ],
        min_sources=0,
    ),
    # =========================================================================
    # Multi-Query Expansion (long complex queries)
    # =========================================================================
    TestScenario(
        id="E50",
        name="Multi-query: very long complex question",
        feature=Feature.MULTI_QUERY,
        query=(
            "I need a comprehensive analysis of TechCorp's product lineup including "
            "all vehicle models with their prices, ranges, battery capacities, and "
            "key distinguishing features. Also include information about the target "
            "market segment for each model."
        ),
        # Should retrieve info about all 3 models
        must_contain_any=["X100", "Y200", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Temporal Queries
    # =========================================================================
    TestScenario(
        id="E51",
        name="Temporal: specific quarter",
        feature=Feature.BASIC_RETRIEVAL,
        query="What happened in Q1 2024 at TechCorp?",
        must_contain_any=["revenue", "stock", "20%", "record"],
        min_sources=1,
    ),
    TestScenario(
        id="E52",
        name="Temporal: founding dates comparison",
        feature=Feature.COMPARISON,
        query="Which company was founded first, TechCorp or GreenDrive?",
        # GreenDrive 2012, TechCorp 2015
        must_contain_any=["GreenDrive", "2012", "first", "earlier", "before"],
        min_sources=1,
    ),
    # =========================================================================
    # Aggregation Queries
    # =========================================================================
    TestScenario(
        id="E53",
        name="Aggregation: count across sources",
        feature=Feature.MULTI_QUERY,
        query="How many different companies are mentioned in the documents?",
        # TechCorp, GreenDrive, AutoMotors, Toyota (mentioned)
        must_contain_any=["TechCorp", "GreenDrive", "AutoMotors", "3", "4", "three", "four"],
        min_sources=1,
    ),
    TestScenario(
        id="E54",
        name="Aggregation: list all products",
        feature=Feature.MULTI_QUERY,
        query="List all vehicle models mentioned across all documents",
        must_contain=["X100"],
        must_contain_any=["Y200", "Z50"],
        min_sources=1,
    ),
    # =========================================================================
    # Cross-Document Synthesis
    # =========================================================================
    TestScenario(
        id="E55",
        name="Cross-doc: info from multiple sources",
        feature=Feature.ENTITY_GRAPH,
        query="What is the complete profile of Sarah Chen including her education and career history?",
        # Info spread across people.md
        must_contain_any=["MIT", "PhD", "AutoMotors", "CEO", "2019"],
        min_sources=1,
    ),
    # =========================================================================
    # Numeric Precision
    # =========================================================================
    TestScenario(
        id="E56",
        name="Numeric: exact number lookup",
        feature=Feature.BASIC_RETRIEVAL,
        query="What is the exact battery capacity of the Model X100 in kWh?",
        must_contain=["75"],
        min_sources=1,
    ),
    TestScenario(
        id="E57",
        name="Numeric: calculation required",
        feature=Feature.COMPARISON,
        query="What is the total battery capacity if you bought one of each TechCorp model?",
        # 75 + 100 + 50 = 225 kWh
        must_contain_any=["225", "75", "100", "50"],
        min_sources=1,
    ),
    # =========================================================================
    # Ambiguous Queries
    # =========================================================================
    TestScenario(
        id="E58",
        name="Ambiguous: which 'company'",
        feature=Feature.BASIC_RETRIEVAL,
        query="When was the company founded?",
        # Ambiguous - could be TechCorp (2015) or GreenDrive (2012)
        must_contain_any=["2015", "2012", "TechCorp", "GreenDrive"],
        min_sources=1,
    ),
    # =========================================================================
    # Code: Edge Cases
    # =========================================================================
    TestScenario(
        id="E59",
        name="Code: method signature",
        feature=Feature.CODE_SEARCH,
        query="What parameters does the register_user method take?",
        must_contain_any=["username", "password", "email", "role"],
        min_sources=1,
    ),
    TestScenario(
        id="E60",
        name="Code: return type",
        feature=Feature.CODE_SEARCH,
        query="What does the login method return?",
        must_contain_any=["token", "str", "string", "session"],
        min_sources=1,
    ),
    # =========================================================================
    # Table: Edge Cases
    # =========================================================================
    TestScenario(
        id="E61",
        name="Table: manager lookup",
        feature=Feature.TABLE_QUERY,
        query="Who is the manager of Alice Wong?",
        # Alice Wong's manager_id is E010, which is James Wilson
        must_contain_any=["James Wilson", "E010", "manager"],
        min_sources=1,
    ),
    TestScenario(
        id="E62",
        name="Table: date-based query",
        feature=Feature.TABLE_QUERY,
        query="Who was hired in 2022?",
        # David Lee (2022-02-20), Nathan Park (2022-06-10)
        must_contain_any=["David Lee", "Nathan Park", "2022"],
        min_sources=1,
    ),
    # =========================================================================
    # Freshness / Authority Boosting
    # =========================================================================
    TestScenario(
        id="E63",
        name="Freshness: official spec keyword",
        feature=Feature.FRESHNESS,
        query="What is the official EPA range for the Model X100?",
        # Official spec says 298 miles (EPA certified), products.md says 300 miles
        must_contain_any=["298", "EPA", "certified", "official"],
        min_sources=1,
    ),
    TestScenario(
        id="E64",
        name="Freshness: spec keyword for warranty",
        feature=Feature.FRESHNESS,
        query="What does the specification say about battery warranty?",
        # Spec document has official warranty terms
        must_contain_any=["8 years", "150,000 miles", "official", "warranty"],
        min_sources=1,
    ),
    TestScenario(
        id="E65",
        name="Freshness: authoritative towing capacity",
        feature=Feature.FRESHNESS,
        query="What is the authoritative maximum towing capacity for Model Y200?",
        # Only in spec document: 5,500 lbs
        must_contain_any=["5,500", "5500", "towing"],
        min_sources=1,
    ),
    TestScenario(
        id="E66",
        name="Freshness: official service requirements",
        feature=Feature.FRESHNESS,
        query="What are the official service requirements for TechCorp vehicles?",
        # Only in spec document
        must_contain_any=["12,500 miles", "12 months", "brake fluid", "service"],
        min_sources=1,
    ),
    # =========================================================================
    # Hybrid Search (Dense + Sparse)
    # =========================================================================
    TestScenario(
        id="E67",
        name="Hybrid: exact model number",
        feature=Feature.HYBRID_SEARCH,
        query="X100 battery capacity",
        # Exact model number should be matched by sparse search
        must_contain_any=["75 kWh", "75kWh", "X100"],
        min_sources=1,
    ),
    TestScenario(
        id="E68",
        name="Hybrid: technical identifier",
        feature=Feature.HYBRID_SEARCH,
        query="CCS Combo 1 charging standard",
        # Technical identifier from spec doc
        must_contain_any=["CCS", "DC Fast", "charging"],
        min_sources=1,
    ),
    TestScenario(
        id="E69",
        name="Hybrid: specific department name",
        feature=Feature.HYBRID_SEARCH,
        query="employees in Engineering",
        # Department name should be exact matched
        must_contain_any=["Engineering", "employee"],
        min_sources=1,
    ),
    TestScenario(
        id="E70",
        name="Hybrid: acronym lookup",
        feature=Feature.HYBRID_SEARCH,
        query="What is SAE Level 2+",
        # Technical acronym from spec
        must_contain_any=["SAE", "Level 2", "autonomous", "driving", "certified"],
        min_sources=1,
    ),
    # =========================================================================
    # Query Expansion (Synonym/Acronym Variations)
    # =========================================================================
    TestScenario(
        id="E71",
        name="Query expansion: synonym for retrieve",
        feature=Feature.QUERY_EXPANSION,
        query="How do I fetch employee data?",
        # "fetch" should expand to "retrieve/get" and find relevant content
        must_contain_any=["employee", "data", "retrieve", "get", "query"],
        min_sources=1,
    ),
    TestScenario(
        id="E72",
        name="Query expansion: synonym for create",
        feature=Feature.QUERY_EXPANSION,
        query="How do I add a new user account?",
        # "add" should expand to "create/register" and find auth module content
        must_contain_any=["register", "create", "user", "account"],
        min_sources=1,
    ),
    TestScenario(
        id="E73",
        name="Query expansion: acronym db to database",
        feature=Feature.QUERY_EXPANSION,
        query="How does the db connection work?",
        # "db" should expand to "database" and find Data Service content
        must_contain_any=["database", "connection", "PostgreSQL", "pool"],
        min_sources=1,
    ),
    TestScenario(
        id="E74",
        name="Query expansion: synonym for error",
        feature=Feature.QUERY_EXPANSION,
        query="What failures can occur in authentication?",
        # "failures" should expand to "errors/exceptions" and find auth content
        must_contain_any=["error", "exception", "authentication", "fail"],
        min_sources=1,
    ),
    # =========================================================================
    # Temporal Queries (Time-Based Comparisons and Periods)
    # =========================================================================
    TestScenario(
        id="E75",
        name="Temporal: Q1 2024 period query",
        feature=Feature.TEMPORAL,
        query="What happened at TechCorp in Q1 2024?",
        # Should retrieve Q1 2024 content from conflicts.md and causal_claims.md
        must_contain_any=["Q1 2024", "revenue", "stock", "20%", "record"],
        min_sources=1,
    ),
    TestScenario(
        id="E76",
        name="Temporal: year-based comparison",
        feature=Feature.TEMPORAL,
        query="Compare TechCorp's founding in 2015 with GreenDrive's founding in 2012",
        # Should find info about both companies and their founding years
        must_contain_any=["2015", "2012", "TechCorp", "GreenDrive", "founded"],
        min_sources=1,
    ),
    TestScenario(
        id="E77",
        name="Temporal: before a specific event",
        feature=Feature.TEMPORAL,
        query="What was Sarah Chen's role before becoming CEO in 2019?",
        # Should find info about her previous role at AutoMotors
        must_contain_any=["AutoMotors", "engineer", "2019", "CEO"],
        min_sources=1,
    ),
    TestScenario(
        id="E78",
        name="Temporal: what changed query",
        feature=Feature.TEMPORAL,
        query="What changed for TechCorp's stock price?",
        # Should detect "what changed" and find stock-related content
        must_contain_any=["stock", "20%", "rose", "price", "TECH"],
        min_sources=1,
    ),
    # =========================================================================
    # Aggregation Queries (List All, Count, Enumerate)
    # =========================================================================
    TestScenario(
        id="E79",
        name="Aggregation: list all people",
        feature=Feature.AGGREGATION,
        query="List all the people mentioned in the company documents",
        # Should fetch more chunks to find all people (Sarah Chen, Marcus Webb, etc.)
        must_contain_any=["Sarah Chen", "Marcus Webb"],
        min_sources=1,
    ),
    TestScenario(
        id="E80",
        name="Aggregation: what are the different companies",
        feature=Feature.AGGREGATION,
        query="What are the different companies mentioned?",
        # Should detect "what are the different" and find all company references
        must_contain_any=["TechCorp", "GreenDrive", "AutoMotors"],
        min_sources=1,
    ),
    TestScenario(
        id="E81",
        name="Aggregation: how many query",
        feature=Feature.AGGREGATION,
        query="How many products does TechCorp Industries have?",
        # Should fetch comprehensive results for accurate count
        must_contain_any=["vehicle", "product", "AI", "autonomous"],
        min_sources=1,
    ),
    TestScenario(
        id="E82",
        name="Aggregation: enumerate features",
        feature=Feature.AGGREGATION,
        query="Enumerate all the features of the autonomous vehicle system",
        # Should list out features comprehensively
        must_contain_any=["autonomous", "AI", "navigation", "sensor"],
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
