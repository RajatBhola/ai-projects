"""Schema mapping: turn the live database schema plus the semantic layer into the
compact context the LLM sees.

For each question it:
1. scores every table for relevance (table names, synonyms, column names, and
   tables referenced by matched business terms),
2. keeps the most relevant tables plus any tables needed to join them,
3. adds column types, descriptions and, for low-cardinality text columns, the
   actual allowed values (so the model writes 'completed', not 'Completed'),
4. appends the business definitions the question mentions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .database import Column, Database
from .semantic import BusinessTerm, SemanticLayer, mentions

TEXT_TYPES = ("CHAR", "TEXT", "STRING", "VARCHAR")
SKIP_VALUE_COLUMNS = {"email", "full_name"}


@dataclass
class SchemaContext:
    text: str
    tables: list[str]
    matched_terms: list[BusinessTerm] = field(default_factory=list)
    dialect: str = "duckdb"


def score_tables(question: str, schema: dict[str, list[Column]], semantic: SemanticLayer,
                 terms: list[BusinessTerm]) -> dict[str, int]:
    scores = {}
    for table, columns in schema.items():
        info = semantic.tables.get(table)
        score = 0
        names = [table, table.replace("_", " ")] + (info.synonyms if info else [])
        if any(mentions(question, n) for n in names):
            score += 3
        score += sum(1 for c in columns if mentions(question, c.name.replace("_", " ")))
        score += sum(2 for t in terms if f"{table}." in t.definition)
        scores[table] = score
    return scores


def _join_path(start: str, goal: str, graph: dict[str, set[str]]) -> list[str]:
    """Shortest list of tables connecting start to goal (BFS over relationships)."""
    prev: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == goal:
            path = []
            while node is not None:
                path.append(node)
                node = prev[node]
            return path
        for nxt in graph.get(node, ()):
            if nxt not in prev:
                prev[nxt] = node
                queue.append(nxt)
    return []


def select_tables(scores: dict[str, int], semantic: SemanticLayer, max_tables: int | None,
                  required: list[str] | None = None) -> list[str]:
    """Pick tables for the prompt. ``required`` tables (used by a matched business
    definition) are always kept; tables needed to join the picks are added on top
    of the budget, because a query can't be written without them."""
    ranked = sorted(scores, key=lambda t: (-scores[t], t))
    if max_tables is None or len(ranked) <= max_tables:
        return ranked
    required = [t for t in (required or []) if t in scores]
    # Always keep at least the single best table the question itself points at.
    extra_budget = max(max_tables - len(required), 1)
    chosen = required + [t for t in ranked if scores[t] > 0 and t not in required][:extra_budget]
    if not chosen:  # nothing matched: let the model see everything
        return ranked
    graph: dict[str, set[str]] = {}
    for left, right, _ in semantic.joins():
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    for other in chosen[1:]:
        for table in _join_path(chosen[0], other, graph):
            if table not in chosen:
                chosen.append(table)
    return chosen


def build_context(db: Database, semantic: SemanticLayer, question: str,
                  max_tables: int | None = None, sample_values: bool = True) -> SchemaContext:
    schema = db.tables()
    terms = semantic.match_terms(question)
    required = [t for t in schema if any(f"{t}." in term.definition for term in terms)]
    tables = select_tables(score_tables(question, schema, semantic, terms), semantic, max_tables, required)

    lines = []
    for table in tables:
        info = semantic.tables.get(table)
        header = f"TABLE {table}"
        if info and info.description:
            header += f"  -- {info.description}"
        lines.append(header)
        for col in schema[table]:
            line = f"  {col.name} {col.type}"
            desc = info.columns.get(col.name) if info else None
            if desc:
                line += f"  -- {desc}"
            if sample_values and col.name not in SKIP_VALUE_COLUMNS and col.type.upper().startswith(TEXT_TYPES):
                values = db.distinct_values(table, col.name)
                if values:
                    line += "  [values: " + ", ".join(f"'{v}'" for v in values) + "]"
            lines.append(line)
        lines.append("")

    rels = [cond for left, right, cond in semantic.joins() if left in tables and right in tables]
    if rels:
        lines.append("JOINS")
        lines.extend(f"  {r}" for r in rels)
        lines.append("")
    if terms:
        lines.append("BUSINESS DEFINITIONS (follow these exactly)")
        lines.extend(f"  {t.name}: {t.definition}" for t in terms)
        lines.append("")

    return SchemaContext(text="\n".join(lines).strip(), tables=tables, matched_terms=terms, dialect=db.dialect)
