"""Deterministic, corpus-derived semantic term graph for retrieval expansion."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Iterable

from fitz_sage.engines.fitz_krag.ingestion.schema import TABLE_PREFIX

if TYPE_CHECKING:
    import sqlite3

    from fitz_sage.storage.sqlite import SqliteConnectionManager


_TERMS = f"{TABLE_PREFIX}semantic_terms"
_FORMS = f"{TABLE_PREFIX}semantic_forms"
_OCCURRENCES = f"{TABLE_PREFIX}semantic_occurrences"
_OBSERVATIONS = f"{TABLE_PREFIX}semantic_relation_observations"
_RELATIONS = f"{TABLE_PREFIX}semantic_relations"
_CLUSTERS = f"{TABLE_PREFIX}semantic_clusters"
_METADATA = f"{TABLE_PREFIX}semantic_metadata"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]*|\d+[A-Za-z0-9._-]*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\r?\n+")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")
_PAREN_ACRONYM_RE = re.compile(r"\(([A-Z][A-Z0-9]{1,9})\)")
_REVERSE_ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9})\s*\(([^()\r\n]{3,120})\)")
_ERROR_RE = re.compile(
    r"\b(?:error(?:\s+code)?|err|http|sqlstate|hresult|cve|kb)"
    r"\s*[:#-]?\s*[A-Z0-9][A-Z0-9._-]{2,}\b",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9]*[_./:-][A-Za-z0-9_./:-]+|"
    r"[a-z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*)\b"
)
_CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9-]+\s+){1,4}[A-Z][A-Za-z0-9-]+\b")
_ALIAS_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Za-z][A-Za-z0-9_.-]*){0,4})"
    r"\s+(also known as|aka|formerly called|renamed to)\s+"
    r"([A-Za-z][A-Za-z0-9_.-]*(?:\s+[A-Za-z][A-Za-z0-9_.-]*){0,4})\b",
    re.IGNORECASE,
)
_IDENT_SPLIT_RE = re.compile(r"[._/\s:-]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "was",
    "were",
    "when",
    "where",
    "which",
    "with",
}

_RELATION_PRIORS = {
    "abbreviation_of": 1.0,
    "alias_of": 0.96,
    "renamed_to": 0.94,
    "identifier_variant_of": 0.90,
    "error_of": 0.84,
    "defined_by": 0.68,
    "cooccurs_with": 0.56,
    "cluster_member": 0.35,
}


@dataclass(frozen=True)
class SemanticExpansion:
    """One explainable query expansion produced by the collection graph."""

    term: str
    score: float
    relation: str
    source: str
    supporting_documents: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Term:
    canonical: str
    kind: str

    @property
    def key(self) -> tuple[str, str]:
        return (_normalize(self.canonical), self.kind)


@dataclass(frozen=True)
class _FormFact:
    term: _Term
    surface: str
    form_type: str
    confidence: float
    extractor: str


@dataclass(frozen=True)
class _RelationFact:
    source: _Term
    target: _Term
    relation_type: str
    confidence: float
    extractor: str


@dataclass
class _UnitFacts:
    key: str
    terms: set[_Term]
    forms: list[_FormFact]
    relations: list[_RelationFact]


class SemanticIndex:
    """Store and query a collection-local semantic term graph in SQLite."""

    def __init__(self, connection_manager: "SqliteConnectionManager", collection: str) -> None:
        self._cm = connection_manager
        self._collection = collection

    def index_file(self, raw_file_id: str, content: str) -> None:
        """Replace one file's observations and mark graph aggregates dirty."""
        units = _extract_units(content)
        with self._cm.connection(self._collection) as conn:
            self._delete_file_observations(conn, raw_file_id)
            term_ids: dict[tuple[str, str], int] = {}

            for unit in units:
                all_terms = set(unit.terms)
                all_terms.update(fact.term for fact in unit.forms)
                for relation in unit.relations:
                    all_terms.add(relation.source)
                    all_terms.add(relation.target)
                for term in all_terms:
                    term_ids[term.key] = self._term_id(conn, term)

                for term in unit.terms:
                    conn.execute(
                        f"""
                        INSERT INTO {_OCCURRENCES}
                            (raw_file_id, unit_key, term_id, occurrence_count)
                        VALUES (?, ?, ?, 1)
                        ON CONFLICT(raw_file_id, unit_key, term_id)
                        DO UPDATE SET occurrence_count = occurrence_count + 1
                        """,
                        (raw_file_id, unit.key, term_ids[term.key]),
                    )

                for fact in unit.forms:
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {_FORMS}
                            (raw_file_id, unit_key, term_id, surface,
                             normalized_surface, form_type, confidence, extractor)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            raw_file_id,
                            unit.key,
                            term_ids[fact.term.key],
                            fact.surface,
                            _normalize(fact.surface),
                            fact.form_type,
                            fact.confidence,
                            fact.extractor,
                        ),
                    )

                for fact in unit.relations:
                    source_id = term_ids[fact.source.key]
                    target_id = term_ids[fact.target.key]
                    if source_id == target_id:
                        continue
                    conn.execute(
                        f"""
                        INSERT OR REPLACE INTO {_OBSERVATIONS}
                            (raw_file_id, unit_key, source_term_id, target_term_id,
                             relation_type, confidence, extractor)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            raw_file_id,
                            unit.key,
                            source_id,
                            target_id,
                            fact.relation_type,
                            fact.confidence,
                            fact.extractor,
                        ),
                    )

            self._mark_dirty(conn)
            conn.commit()

    def remove_file(self, raw_file_id: str) -> None:
        """Remove one file's graph evidence before its source row is deleted."""
        with self._cm.connection(self._collection) as conn:
            self._delete_file_observations(conn, raw_file_id)
            self._mark_dirty(conn)
            conn.commit()

    def expand(self, query: str, *, max_terms: int = 6) -> list[SemanticExpansion]:
        """Return context-ranked, collection-backed expansions for a query."""
        if max_terms <= 0 or not query.strip():
            return []
        query_phrases = _query_phrases(query)
        if not query_phrases:
            return []

        with self._cm.connection(self._collection) as conn:
            self._materialize_if_dirty(conn)
            matches = self._matched_terms(conn, query_phrases)
            if not matches:
                return []
            candidates = self._form_candidates(conn, matches, query_phrases)
            self._relation_candidates(conn, matches, query_phrases, candidates)
            self._cluster_candidates(conn, matches, query_phrases, candidates)

        ordered = sorted(
            candidates.values(),
            key=lambda item: (-item.score, item.term.casefold()),
        )
        return ordered[:max_terms]

    def stats(self) -> dict[str, int]:
        """Return inspectable graph counts, materializing pending aggregates."""
        with self._cm.connection(self._collection) as conn:
            self._materialize_if_dirty(conn)
            return {
                "terms": _count(conn, _TERMS, "document_frequency > 0"),
                "forms": _count(conn, _FORMS),
                "observations": _count(conn, _OBSERVATIONS),
                "relations": _count(conn, _RELATIONS),
                "clusters": int(
                    conn.execute(f"SELECT COUNT(DISTINCT cluster_id) FROM {_CLUSTERS}").fetchone()[
                        0
                    ]
                ),
            }

    @staticmethod
    def _delete_file_observations(conn: "sqlite3.Connection", raw_file_id: str) -> None:
        for table in (_FORMS, _OCCURRENCES, _OBSERVATIONS):
            conn.execute(f"DELETE FROM {table} WHERE raw_file_id = ?", (raw_file_id,))

    @staticmethod
    def _term_id(conn: "sqlite3.Connection", term: _Term) -> int:
        normalized, kind = term.key
        conn.execute(
            f"""
            INSERT INTO {_TERMS} (canonical, normalized, kind)
            VALUES (?, ?, ?)
            ON CONFLICT(normalized, kind) DO NOTHING
            """,
            (term.canonical.strip(), normalized, kind),
        )
        row = conn.execute(
            f"SELECT id FROM {_TERMS} WHERE normalized = ? AND kind = ?",
            (normalized, kind),
        ).fetchone()
        assert row is not None
        return int(row[0])

    @staticmethod
    def _mark_dirty(conn: "sqlite3.Connection") -> None:
        conn.execute(
            f"""
            INSERT INTO {_METADATA} (key, value) VALUES ('dirty', '1')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )

    def _materialize_if_dirty(self, conn: "sqlite3.Connection") -> None:
        row = conn.execute(f"SELECT value FROM {_METADATA} WHERE key = 'dirty'").fetchone()
        if row is None or row[0] == "1":
            self._materialize(conn)

    def _materialize(self, conn: "sqlite3.Connection") -> None:
        conn.execute(
            f"""
            UPDATE {_TERMS}
            SET document_frequency = (
                SELECT COUNT(DISTINCT raw_file_id)
                FROM {_OCCURRENCES} occurrence
                WHERE occurrence.term_id = {_TERMS}.id
            )
            """
        )
        conn.execute(f"DELETE FROM {_RELATIONS}")
        rows = conn.execute(
            f"""
            SELECT source_term_id, target_term_id, relation_type,
                   MAX(confidence), COUNT(DISTINCT raw_file_id),
                   COUNT(DISTINCT raw_file_id || ':' || unit_key),
                   GROUP_CONCAT(DISTINCT extractor)
            FROM {_OBSERVATIONS}
            GROUP BY source_term_id, target_term_id, relation_type
            """
        ).fetchall()
        for source, target, relation, confidence, documents, units, extractors in rows:
            support_bonus = 0.06 * math.log2(max(1, int(units)))
            weight = min(0.99, float(confidence) + support_bonus)
            conn.execute(
                f"""
                INSERT INTO {_RELATIONS}
                    (source_term_id, target_term_id, relation_type, weight,
                     supporting_documents, supporting_units, extractor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source, target, relation, weight, documents, units, extractors or ""),
            )
        self._rebuild_clusters(conn)
        conn.execute(
            f"""
            INSERT INTO {_METADATA} (key, value) VALUES ('dirty', '0')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
        conn.execute(
            f"""
            INSERT INTO {_METADATA} (key, value) VALUES ('built_at', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()

    @staticmethod
    def _rebuild_clusters(conn: "sqlite3.Connection") -> None:
        conn.execute(f"DELETE FROM {_CLUSTERS}")
        rows = conn.execute(
            f"""
            SELECT source_term_id, target_term_id, weight, relation_type
            FROM {_RELATIONS}
            WHERE weight >= 0.62 OR relation_type != 'cooccurs_with'
            """
        ).fetchall()
        parent: dict[int, int] = {}
        membership: dict[int, float] = defaultdict(float)

        def find(value: int) -> int:
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for source, target, weight, _relation in rows:
            source_id, target_id = int(source), int(target)
            union(source_id, target_id)
            membership[source_id] = max(membership[source_id], float(weight))
            membership[target_id] = max(membership[target_id], float(weight))

        groups: dict[int, list[int]] = defaultdict(list)
        for term_id in parent:
            groups[find(term_id)].append(term_id)
        for members in groups.values():
            if len(members) < 2:
                continue
            cluster_id = min(members)
            for term_id in members:
                conn.execute(
                    f"INSERT INTO {_CLUSTERS} (cluster_id, term_id, membership) VALUES (?, ?, ?)",
                    (cluster_id, term_id, membership.get(term_id, 0.62)),
                )

    @staticmethod
    def _matched_terms(
        conn: "sqlite3.Connection", query_phrases: set[str]
    ) -> dict[int, tuple[str, str, float, int]]:
        placeholders = ",".join("?" for _ in query_phrases)
        params = tuple(query_phrases)
        matches: dict[int, tuple[str, str, float, int]] = {}
        rows = conn.execute(
            f"""
            SELECT term.id, term.canonical, form.form_type,
                   MAX(form.confidence), COUNT(DISTINCT form.raw_file_id)
            FROM {_FORMS} form
            JOIN {_TERMS} term ON term.id = form.term_id
            WHERE form.normalized_surface IN ({placeholders})
              AND term.document_frequency > 0
            GROUP BY term.id, term.canonical, form.form_type
            """,
            params,
        ).fetchall()
        for term_id, canonical, form_type, confidence, documents in rows:
            matches[int(term_id)] = (
                str(canonical),
                str(form_type),
                float(confidence),
                int(documents),
            )

        rows = conn.execute(
            f"""
            SELECT id, canonical, document_frequency
            FROM {_TERMS}
            WHERE normalized IN ({placeholders}) AND document_frequency > 0
            """,
            params,
        ).fetchall()
        for term_id, canonical, documents in rows:
            matches.setdefault(
                int(term_id),
                (str(canonical), "literal", 1.0, int(documents)),
            )
        return matches

    @staticmethod
    def _form_candidates(
        conn: "sqlite3.Connection",
        matches: dict[int, tuple[str, str, float, int]],
        query_phrases: set[str],
    ) -> dict[str, SemanticExpansion]:
        candidates: dict[str, SemanticExpansion] = {}
        placeholders = ",".join("?" for _ in matches)
        rows = conn.execute(
            f"""
            SELECT form.term_id, form.surface, form.normalized_surface,
                   form.form_type, MAX(form.confidence),
                   COUNT(DISTINCT form.raw_file_id)
            FROM {_FORMS} form
            WHERE form.term_id IN ({placeholders})
            GROUP BY form.term_id, form.surface, form.normalized_surface, form.form_type
            """,
            tuple(matches),
        ).fetchall()
        for term_id, surface, normalized, form_type, confidence, documents in rows:
            if str(normalized) in query_phrases or str(form_type) == "identity":
                continue
            source = matches[int(term_id)][0]
            _add_candidate(
                candidates,
                SemanticExpansion(
                    term=str(surface),
                    score=round(float(confidence), 6),
                    relation=str(form_type),
                    source=source,
                    supporting_documents=int(documents),
                ),
            )

        for canonical, form_type, confidence, documents in matches.values():
            if _normalize(canonical) in query_phrases or form_type in {"identity", "literal"}:
                continue
            _add_candidate(
                candidates,
                SemanticExpansion(
                    term=canonical,
                    score=round(confidence, 6),
                    relation=form_type,
                    source=canonical,
                    supporting_documents=documents,
                ),
            )
        return candidates

    @staticmethod
    def _relation_candidates(
        conn: "sqlite3.Connection",
        matches: dict[int, tuple[str, str, float, int]],
        query_phrases: set[str],
        candidates: dict[str, SemanticExpansion],
    ) -> None:
        placeholders = ",".join("?" for _ in matches)
        params = (*matches, *matches)
        rows = conn.execute(
            f"""
            SELECT relation.source_term_id, relation.target_term_id,
                   relation.relation_type, relation.weight,
                   relation.supporting_documents,
                   source.canonical, target.canonical,
                   source.document_frequency, target.document_frequency
            FROM {_RELATIONS} relation
            JOIN {_TERMS} source ON source.id = relation.source_term_id
            JOIN {_TERMS} target ON target.id = relation.target_term_id
            WHERE relation.source_term_id IN ({placeholders})
               OR relation.target_term_id IN ({placeholders})
            """,
            params,
        ).fetchall()
        connections: Counter[int] = Counter()
        pending: list[tuple[int, str, str, float, int]] = []
        for (
            source_id,
            target_id,
            relation,
            weight,
            documents,
            source,
            target,
            source_df,
            target_df,
        ) in rows:
            source_id, target_id = int(source_id), int(target_id)
            if source_id in matches:
                candidate_id, candidate, candidate_df = target_id, str(target), int(target_df)
                matched_source = matches[source_id][0]
            else:
                candidate_id, candidate, candidate_df = source_id, str(source), int(source_df)
                matched_source = matches[target_id][0]
            if (
                candidate_id in matches
                or candidate_df <= 0
                or _normalize(candidate) in query_phrases
            ):
                continue
            connections[candidate_id] += 1
            prior = _RELATION_PRIORS.get(str(relation), 0.5)
            score = float(weight) * prior
            pending.append((candidate_id, candidate, str(relation), score, int(documents)))
            _add_candidate(
                candidates,
                SemanticExpansion(
                    term=candidate,
                    score=round(score, 6),
                    relation=str(relation),
                    source=matched_source,
                    supporting_documents=int(documents),
                ),
            )

        for candidate_id, candidate, relation, score, documents in pending:
            if connections[candidate_id] <= 1:
                continue
            key = _normalize(candidate)
            existing = candidates.get(key)
            if existing is None or existing.relation != relation:
                continue
            candidates[key] = SemanticExpansion(
                term=existing.term,
                score=round(min(0.99, score + 0.08 * (connections[candidate_id] - 1)), 6),
                relation=existing.relation,
                source=existing.source,
                supporting_documents=max(existing.supporting_documents, documents),
            )

    @staticmethod
    def _cluster_candidates(
        conn: "sqlite3.Connection",
        matches: dict[int, tuple[str, str, float, int]],
        query_phrases: set[str],
        candidates: dict[str, SemanticExpansion],
    ) -> None:
        placeholders = ",".join("?" for _ in matches)
        rows = conn.execute(
            f"""
            SELECT DISTINCT member.term_id, term.canonical, member.membership,
                            term.document_frequency
            FROM {_CLUSTERS} seed
            JOIN {_CLUSTERS} member ON member.cluster_id = seed.cluster_id
            JOIN {_TERMS} term ON term.id = member.term_id
            WHERE seed.term_id IN ({placeholders})
            """,
            tuple(matches),
        ).fetchall()
        source = next(iter(matches.values()))[0]
        for term_id, canonical, membership, documents in rows:
            if int(term_id) in matches or int(documents) <= 0:
                continue
            if _normalize(str(canonical)) in query_phrases:
                continue
            score = float(membership) * _RELATION_PRIORS["cluster_member"]
            _add_candidate(
                candidates,
                SemanticExpansion(
                    term=str(canonical),
                    score=round(score, 6),
                    relation="cluster_member",
                    source=source,
                    supporting_documents=int(documents),
                ),
            )


def _extract_units(content: str) -> list[_UnitFacts]:
    raw_units = [part.strip() for part in _SENTENCE_SPLIT_RE.split(content) if part.strip()]
    phrase_counts = _document_phrase_counts(raw_units)
    units: list[_UnitFacts] = []
    for index, text in enumerate(raw_units):
        unit_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        units.append(_extract_unit(text, f"{index}:{unit_hash}", phrase_counts))
    return units


def _extract_unit(text: str, unit_key: str, phrase_counts: Counter[str]) -> _UnitFacts:
    terms: set[_Term] = set()
    strong_terms: set[_Term] = set()
    forms: list[_FormFact] = []
    explicit_relations: list[_RelationFact] = []

    def add_term(
        value: str,
        kind: str,
        *,
        surface: str | None = None,
        form_type: str = "identity",
        confidence: float = 1.0,
        extractor: str = "lexical",
    ) -> _Term | None:
        cleaned = _clean_surface(value)
        if not _useful_term(cleaned):
            return None
        term = _Term(cleaned, kind)
        terms.add(term)
        forms.append(
            _FormFact(
                term=term,
                surface=_clean_surface(surface or value),
                form_type=form_type,
                confidence=confidence,
                extractor=extractor,
            )
        )
        return term

    acronym_pairs = _acronym_pairs(text)
    for long_form, acronym in acronym_pairs:
        long_term = add_term(long_form, "phrase", extractor="parenthetical_acronym")
        short_term = add_term(acronym, "acronym", extractor="parenthetical_acronym")
        if long_term and short_term:
            strong_terms.update((long_term, short_term))
            forms.append(
                _FormFact(
                    term=long_term,
                    surface=acronym,
                    form_type="abbreviation",
                    confidence=0.99,
                    extractor="parenthetical_acronym",
                )
            )
            explicit_relations.append(
                _RelationFact(
                    source=short_term,
                    target=long_term,
                    relation_type="abbreviation_of",
                    confidence=0.99,
                    extractor="parenthetical_acronym",
                )
            )

    for match in _ALIAS_RE.finditer(text):
        left = add_term(match.group(1), "phrase", extractor="explicit_alias")
        right = add_term(match.group(3), "phrase", extractor="explicit_alias")
        if left and right:
            strong_terms.update((left, right))
            relation = "renamed_to" if "renamed" in match.group(2).casefold() else "alias_of"
            explicit_relations.append(_RelationFact(left, right, relation, 0.96, "explicit_alias"))

    error_terms: set[_Term] = set()
    for match in _ERROR_RE.finditer(text):
        term = add_term(match.group(0), "error", extractor="error_identifier")
        if term:
            error_terms.add(term)
            strong_terms.add(term)

    for match in _ACRONYM_RE.finditer(text):
        term = add_term(match.group(0), "acronym", extractor="acronym_token")
        if term:
            strong_terms.add(term)

    for match in _CAPITALIZED_PHRASE_RE.finditer(text):
        term = add_term(match.group(0), "phrase", extractor="capitalized_phrase")
        if term:
            strong_terms.add(term)

    for match in _IDENTIFIER_RE.finditer(text):
        surface = match.group(0)
        term = add_term(surface, "identifier", extractor="identifier")
        if term:
            strong_terms.add(term)
        split = _clean_surface(" ".join(_IDENT_SPLIT_RE.split(surface)))
        if term and _normalize(split) != _normalize(surface) and _useful_term(split):
            forms.append(
                _FormFact(
                    term=term,
                    surface=split,
                    form_type="identifier_variant",
                    confidence=0.92,
                    extractor="identifier",
                )
            )

    anchored = bool(acronym_pairs or error_terms)
    phrases = _candidate_phrases(text)
    ranked_phrases = sorted(
        phrases,
        key=lambda phrase: (phrase_counts[phrase], len(phrase.split()), len(phrase)),
        reverse=True,
    )
    for phrase in ranked_phrases[:12]:
        if phrase_counts[phrase] >= 2 or anchored:
            add_term(phrase, "phrase", extractor="corpus_phrase")

    ordered_terms = sorted(terms, key=lambda term: (term.kind, term.key))[:20]
    relations = list(explicit_relations)
    explicit_pairs = {
        frozenset((relation.source.key, relation.target.key)) for relation in explicit_relations
    }
    for index, source in enumerate(ordered_terms):
        for target in ordered_terms[index + 1 :]:
            if source.key == target.key:
                continue
            pair = frozenset((source.key, target.key))
            if pair in explicit_pairs:
                continue
            if source in error_terms or target in error_terms:
                relation_type = "error_of"
                context_term = target if source in error_terms else source
                confidence = 0.82 if context_term in strong_terms else 0.55
                extractor = "error_context"
            else:
                relation_type = "cooccurs_with"
                confidence = 0.42
                extractor = "sentence_cooccurrence"
            relations.append(_RelationFact(source, target, relation_type, confidence, extractor))

    return _UnitFacts(unit_key, set(ordered_terms), forms, relations)


def _acronym_pairs(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in _PAREN_ACRONYM_RE.finditer(text):
        acronym = match.group(1)
        prefix = text[max(0, match.start() - 140) : match.start()]
        long_form = _matching_long_form(prefix, acronym)
        if long_form:
            pairs.append((long_form, acronym))
    for match in _REVERSE_ACRONYM_RE.finditer(text):
        acronym, candidate = match.group(1), _clean_surface(match.group(2))
        if _initials(candidate) == _acronym_letters(acronym):
            pairs.append((candidate, acronym))
    return list(dict.fromkeys(pairs))


def _matching_long_form(prefix: str, acronym: str) -> str | None:
    words = _WORD_RE.findall(prefix)
    target = _acronym_letters(acronym)
    for size in range(2, min(10, len(words)) + 1):
        candidate = words[-size:]
        meaningful = [word for word in candidate if word.casefold() not in _STOP_WORDS]
        if "".join(word[0].upper() for word in meaningful) == target:
            return " ".join(candidate)
    return None


def _initials(value: str) -> str:
    words = [word for word in _WORD_RE.findall(value) if word.casefold() not in _STOP_WORDS]
    return "".join(word[0].upper() for word in words)


def _acronym_letters(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _document_phrase_counts(units: Iterable[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for unit in units:
        counts.update(_candidate_phrases(unit))
    return counts


def _candidate_phrases(text: str) -> set[str]:
    tokens = [token for token in _WORD_RE.findall(text) if token.casefold() not in _STOP_WORDS]
    phrases: set[str] = set()
    for size in (2, 3):
        for index in range(len(tokens) - size + 1):
            window = tokens[index : index + size]
            if any(len(token) <= 2 and not token.isdigit() for token in window):
                continue
            phrases.add(_clean_surface(" ".join(window)))
    return phrases


def _query_phrases(query: str) -> set[str]:
    normalized = _normalize(query)
    tokens = normalized.split()
    phrases: set[str] = set()
    for size in range(1, min(5, len(tokens)) + 1):
        for index in range(len(tokens) - size + 1):
            phrases.add(" ".join(tokens[index : index + size]))
    return phrases


def _clean_surface(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,;:.-")


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = " ".join(part for part in _IDENT_SPLIT_RE.split(normalized) if part)
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _useful_term(value: str) -> bool:
    normalized = _normalize(value)
    if not normalized or normalized in _STOP_WORDS or len(normalized) < 2:
        return False
    return len(normalized.split()) <= 6 and len(normalized) <= 120


def _add_candidate(candidates: dict[str, SemanticExpansion], candidate: SemanticExpansion) -> None:
    key = _normalize(candidate.term)
    existing = candidates.get(key)
    if existing is None or candidate.score > existing.score:
        candidates[key] = candidate


def _count(conn: "sqlite3.Connection", table: str, where: str | None = None) -> int:
    clause = f" WHERE {where}" if where else ""
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}{clause}").fetchone()[0])


__all__ = ["SemanticExpansion", "SemanticIndex"]
