# tests/unit/property/strategies.py
"""
Custom Hypothesis strategies for fitz-ai property tests.

Provides domain-specific strategies for generating:
- Query text (realistic search queries)
- Document text (for chunking)
- Keywords with categories
- Chunker parameters
"""

from hypothesis import strategies as st

# =============================================================================
# Query Text Strategies
# =============================================================================

# Common query words (realistic search patterns)
QUERY_WORDS = [
    "how",
    "what",
    "where",
    "when",
    "why",
    "who",
    "find",
    "show",
    "get",
    "list",
    "explain",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "to",
    "from",
    "with",
    "for",
    "in",
    "on",
    "at",
    "error",
    "bug",
    "issue",
    "problem",
    "fix",
    "create",
    "delete",
    "update",
    "add",
    "remove",
    "user",
    "file",
    "data",
    "system",
    "config",
    "api",
    "database",
    "server",
    "client",
]


@st.composite
def query_text(draw, min_words: int = 1, max_words: int = 15) -> str:
    """
    Generate realistic query-like text.

    Examples:
        "how to fix error"
        "find user data"
        "what is the api config"
    """
    num_words = draw(st.integers(min_value=min_words, max_value=max_words))
    words = draw(st.lists(st.sampled_from(QUERY_WORDS), min_size=num_words, max_size=num_words))
    return " ".join(words)


@st.composite
def document_text(draw, min_chars: int = 100, max_chars: int = 5000) -> str:
    """
    Generate document-like text for chunking tests.

    Produces text with:
    - Multiple paragraphs (separated by \\n\\n)
    - Sentences (ending with . )
    - Variable length

    Uses a simpler strategy to avoid excessive entropy consumption.
    """
    # Use a fixed word pool to reduce entropy
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "data",
        "system",
        "process",
        "function",
        "method",
        "class",
        "object",
        "user",
        "input",
        "output",
        "result",
        "value",
        "type",
        "code",
        "test",
    ]

    # Target length with some variance
    target_len = draw(st.integers(min_value=min_chars, max_value=min(max_chars, 2000)))
    result = []
    current_len = 0

    # Build paragraphs until we reach target length
    while current_len < target_len:
        # Build a paragraph (3-8 sentences)
        sentences = []
        for _ in range(draw(st.integers(min_value=3, max_value=8))):
            # Build a sentence (5-12 words)
            num_words = draw(st.integers(min_value=5, max_value=12))
            sentence_words = [draw(st.sampled_from(words)) for _ in range(num_words)]
            sentences.append(" ".join(sentence_words) + ".")

        paragraph = " ".join(sentences)
        result.append(paragraph)
        current_len += len(paragraph) + 2  # +2 for \n\n

    text = "\n\n".join(result)

    # Trim if too long
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


# =============================================================================
# Synonym/Acronym Query Strategies
# =============================================================================

# Known synonyms from expansion.py
KNOWN_SYNONYMS = [
    "delete",
    "remove",
    "create",
    "add",
    "update",
    "modify",
    "get",
    "retrieve",
    "fetch",
    "error",
    "failure",
    "issue",
    "bug",
    "start",
    "stop",
    "run",
    "execute",
    "install",
    "setup",
    "configure",
    "file",
    "document",
    "folder",
    "directory",
    "user",
    "function",
    "method",
    "class",
    "list",
    "array",
    "api",
    "endpoint",
    "database",
    "db",
    "server",
    "client",
    "request",
    "response",
    "enable",
    "disable",
    "active",
    "inactive",
]

# Known acronyms from expansion.py
KNOWN_ACRONYMS = [
    "api",
    "ui",
    "ux",
    "db",
    "sql",
    "html",
    "css",
    "js",
    "ts",
    "url",
    "http",
    "https",
    "json",
    "xml",
    "csv",
    "pdf",
    "id",
    "auth",
    "config",
    "env",
    "dev",
    "prod",
    "repo",
    "pr",
    "ci",
    "cd",
    "k8s",
    "aws",
    "gcp",
    "vm",
    "os",
    "cpu",
    "gpu",
    "ram",
    "ssd",
    "hdd",
    "iot",
    "ml",
    "ai",
    "nlp",
    "llm",
    "rag",
]


@st.composite
def query_with_synonym(draw) -> str:
    """
    Generate a query containing at least one known synonym.

    Examples:
        "delete the file"
        "how to create user"
    """
    synonym = draw(st.sampled_from(KNOWN_SYNONYMS))
    prefix_words = draw(st.integers(min_value=0, max_value=3))
    suffix_words = draw(st.integers(min_value=0, max_value=3))

    prefix = draw(
        st.lists(st.sampled_from(QUERY_WORDS), min_size=prefix_words, max_size=prefix_words)
    )
    suffix = draw(
        st.lists(st.sampled_from(QUERY_WORDS), min_size=suffix_words, max_size=suffix_words)
    )

    words = prefix + [synonym] + suffix
    return " ".join(words)


@st.composite
def query_with_acronym(draw) -> str:
    """
    Generate a query containing at least one known acronym.

    Examples:
        "the api endpoint"
        "how to use sql"
    """
    acronym = draw(st.sampled_from(KNOWN_ACRONYMS))
    prefix_words = draw(st.integers(min_value=0, max_value=3))
    suffix_words = draw(st.integers(min_value=0, max_value=3))

    prefix = draw(
        st.lists(st.sampled_from(QUERY_WORDS), min_size=prefix_words, max_size=prefix_words)
    )
    suffix = draw(
        st.lists(st.sampled_from(QUERY_WORDS), min_size=suffix_words, max_size=suffix_words)
    )

    words = prefix + [acronym] + suffix
    return " ".join(words)


# =============================================================================
# Chunker Parameter Strategies
# =============================================================================


@st.composite
def chunk_params(draw, min_size: int = 10, max_size: int = 5000) -> tuple[int, int]:
    """
    Generate valid (chunk_size, chunk_overlap) pairs.

    Constraints:
        - chunk_size >= 1
        - chunk_overlap >= 0
        - chunk_overlap < chunk_size

    Returns:
        Tuple of (chunk_size, chunk_overlap)
    """
    chunk_size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Overlap must be less than size
    max_overlap = max(0, chunk_size - 1)
    chunk_overlap = draw(st.integers(min_value=0, max_value=max_overlap))
    return (chunk_size, chunk_overlap)


# =============================================================================
# Keyword Strategies (for vocabulary tests)
# =============================================================================


@st.composite
def testcase_id(draw) -> str:
    """
    Generate TC-style test case IDs.

    Examples:
        "TC-1001", "TC_2345", "tc-9999"
    """
    separator = draw(st.sampled_from(["-", "_", ""]))
    number = draw(st.integers(min_value=1, max_value=99999))
    prefix = draw(st.sampled_from(["TC", "tc", "Tc"]))
    return f"{prefix}{separator}{number}"


@st.composite
def ticket_id(draw) -> str:
    """
    Generate JIRA-style ticket IDs.

    Examples:
        "JIRA-123", "PROJ-4567", "ABC_890"
    """
    # 2-5 uppercase letters
    prefix_len = draw(st.integers(min_value=2, max_value=5))
    prefix = "".join(
        draw(
            st.lists(
                st.sampled_from("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                min_size=prefix_len,
                max_size=prefix_len,
            )
        )
    )
    separator = draw(st.sampled_from(["-", "_"]))
    number = draw(st.integers(min_value=1, max_value=99999))
    return f"{prefix}{separator}{number}"


@st.composite
def version_string(draw) -> str:
    """
    Generate version strings.

    Examples:
        "v1.2.3", "2.0.0", "v3.14.159-alpha"
    """
    major = draw(st.integers(min_value=0, max_value=99))
    minor = draw(st.integers(min_value=0, max_value=99))
    patch = draw(st.integers(min_value=0, max_value=99))

    has_v = draw(st.booleans())
    has_suffix = draw(st.booleans())

    version = f"{major}.{minor}.{patch}"

    if has_suffix:
        suffix = draw(st.sampled_from(["alpha", "beta", "rc1", "rc2", "dev"]))
        version = f"{version}-{suffix}"

    if has_v:
        version = f"v{version}"

    return version


@st.composite
def keyword_with_category(draw) -> tuple[str, str]:
    """
    Generate a (keyword, category) pair.

    Returns:
        Tuple of (keyword, category) where category is one of:
        testcase, ticket, version, person, generic
    """
    category = draw(st.sampled_from(["testcase", "ticket", "version", "person", "generic"]))

    if category == "testcase":
        keyword = draw(testcase_id())
    elif category == "ticket":
        keyword = draw(ticket_id())
    elif category == "version":
        keyword = draw(version_string())
    elif category == "person":
        first_names = ["John", "Jane", "Bob", "Alice", "Charlie", "Diana"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia"]
        first = draw(st.sampled_from(first_names))
        last = draw(st.sampled_from(last_names))
        keyword = f"{first} {last}"
    else:  # generic
        keyword = draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
                min_size=3,
                max_size=20,
            )
        )

    return (keyword, category)


# =============================================================================
# General Text Strategies
# =============================================================================


@st.composite
def non_empty_text(draw, min_size: int = 1, max_size: int = 100) -> str:
    """
    Generate non-empty printable text.
    """
    return (
        draw(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
                min_size=min_size,
                max_size=max_size,
            )
        ).strip()
        or "fallback"
    )


# Strategy aliases for easy import
queries = query_text()
documents = document_text()
synonym_queries = query_with_synonym()
acronym_queries = query_with_acronym()
chunker_params = chunk_params()
