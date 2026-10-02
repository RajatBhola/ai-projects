"""Explain a SQL query in plain English.

Pipeline:  SQL -> analyzer (parser facts + issue checks) -> LLM writes the prose,
grounded in those facts and the semantic layer -> output columns are checked
against the parser's list so the model can't invent or drop any.

Without an LLM (``offline=True``) the analyzer's template description is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .analyzer import QueryAnalysis, analyze, describe
from .llm import JSONLLMClient
from .semantic import SemanticLayer

AUDIENCES = {
    "business": (
        "The reader is a non-technical business user. Avoid SQL jargon: say 'combines customers "
        "with their orders' instead of 'inner joins', 'for each country' instead of 'group by'. "
        "Talk about the business meaning (customers, orders, money) using the table descriptions."
    ),
    "technical": (
        "The reader is an analyst or engineer. You may use SQL terms, and mention join types, "
        "grain (what one output row represents) and anything that affects correctness or performance."
    ),
}

SYSTEM_PROMPT = """You explain SQL queries in clear, simple English.

{audience}

Rules:
- Describe only what the query actually does. The STRUCTURE section comes from a SQL parser and is
  reliable: never contradict it, and never claim the query does something that isn't in it.
- Translate codes and columns into meaning using the TABLE DESCRIPTIONS when available
  (e.g. status = 'completed' -> "only completed orders").
- Say what one row of the result represents (for example "one row per country").
- If the query calculates something covered by a BUSINESS DEFINITION, say so; if it deviates from
  that definition (for example counts cancelled orders as revenue), add a caveat.
- DETECTED ISSUES are already shown to the user separately: do not repeat them in caveats.
  Use caveats only for other genuine points (assumptions, surprises, ambiguity). Empty is fine.
- If it changes data (INSERT/UPDATE/DELETE/DDL), make that unmistakable in the summary.
- Keep the summary to one or two sentences and each step to one sentence.

Respond with JSON only."""

USER_PROMPT = """SQL ({dialect})
{sql}

STRUCTURE (from the parser)
{structure}

TABLE DESCRIPTIONS
{tables}

BUSINESS DEFINITIONS
{terms}

DETECTED ISSUES
{issues}"""

EXPLAIN_SCHEMA = {
    "name": "sql_explanation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "What the query does, in one or two sentences."},
            "steps": {"type": "array", "items": {"type": "string"},
                      "description": "What happens, in order, one sentence each."},
            "output_columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "meaning": {"type": "string"}},
                    "required": ["name", "meaning"],
                    "additionalProperties": False,
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "steps", "output_columns", "caveats"],
        "additionalProperties": False,
    },
}


@dataclass
class ColumnMeaning:
    name: str
    meaning: str


@dataclass
class Explanation:
    analysis: QueryAnalysis
    summary: str
    steps: list[str]
    output_columns: list[ColumnMeaning] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    source: str = "llm"          # "llm" or "offline"
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "steps": self.steps,
            "output_columns": [vars(c) for c in self.output_columns],
            "caveats": self.caveats,
            "issues": [vars(i) for i in self.analysis.issues],
            "statement_type": self.analysis.statement_type,
            "modifies_data": self.analysis.modifies_data,
            "tables": self.analysis.tables,
            "sql": self.analysis.sql,
            "source": self.source,
            "tokens": {"input": self.input_tokens, "output": self.output_tokens},
        }


# ----------------------------------------------------------------------------- context
def structure_summary(a: QueryAnalysis) -> str:
    facts = {
        "statement_type": a.statement_type,
        "modifies_data": a.modifies_data,
        "target_table": a.target_table,
        "tables": a.tables,
        "ctes": a.ctes,
        "reads_from": a.source,
        "joins": [vars(j) for j in a.joins],
        "filters": a.filters,
        "set_columns": a.set_columns,
        "group_by": a.group_by,
        "aggregations": a.aggregations,
        "having": a.having,
        "window_functions": a.window_functions,
        "distinct": a.distinct,
        "order_by": a.order_by,
        "limit": a.limit,
        "output_columns": [vars(c) for c in a.output_columns],
        "subqueries": a.subqueries,
        "union_branches": a.union_branches,
    }
    return json.dumps({k: v for k, v in facts.items() if v not in (None, [], "", 0, False)
                       or k in ("statement_type", "modifies_data")}, indent=2)


def table_descriptions(a: QueryAnalysis, semantic: SemanticLayer) -> str:
    lines = []
    for name in a.tables:
        info = semantic.tables.get(name.split(".")[-1])
        if info is None:
            continue
        lines.append(f"{info.name}: {info.description}")
        lines.extend(f"  {col}: {desc}" for col, desc in info.columns.items())
    return "\n".join(lines) or "(none available)"


def relevant_terms(a: QueryAnalysis, semantic: SemanticLayer) -> str:
    names = {t.split(".")[-1] for t in a.tables}
    terms = [t for t in semantic.business_terms.values()
             if any(f"{n}." in t.definition for n in names)]
    return "\n".join(f"{t.name}: {t.definition}" for t in terms) or "(none)"


def build_messages(a: QueryAnalysis, semantic: SemanticLayer, audience: str = "business") -> list[dict]:
    issues = "\n".join(f"- [{i.severity}] {i.message}" for i in a.issues) or "(none)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(audience=AUDIENCES[audience])},
        {"role": "user", "content": USER_PROMPT.format(
            dialect=a.dialect, sql=a.sql, structure=structure_summary(a),
            tables=table_descriptions(a, semantic), terms=relevant_terms(a, semantic), issues=issues,
        )},
    ]


# ----------------------------------------------------------------------------- explain
def ground_columns(a: QueryAnalysis, proposed: list[dict]) -> list[ColumnMeaning]:
    """Use the parser's output columns, filling in the model's meanings by name.

    The model can't add, drop or rename columns. With SELECT * the real column list
    isn't knowable from the SQL alone, so the model's description of * is kept."""
    meanings = {str(c.get("name", "")).strip().lower(): str(c.get("meaning", "")).strip() for c in proposed}
    out = []
    for col in a.output_columns:
        meaning = meanings.get(col.name.lower()) or meanings.get(col.expression.lower(), "")
        if not meaning and col.name == "*":
            meaning = "Every column from the tables read."
        out.append(ColumnMeaning(col.name, meaning))
    return out


def explain_offline(a: QueryAnalysis) -> Explanation:
    steps = describe(a)
    kind = "changes data" if a.modifies_data else "is read-only"
    tables = ", ".join(a.tables) or "no tables"
    summary = f"A {a.statement_type} statement that {kind}, using {tables}."
    return Explanation(analysis=a, summary=summary, steps=steps,
                       output_columns=[ColumnMeaning(c.name, c.expression if c.expression != c.name else "")
                                       for c in a.output_columns],
                       source="offline")


def explain_analysis(a: QueryAnalysis, llm: JSONLLMClient | None, semantic: SemanticLayer | None = None,
                     audience: str = "business") -> Explanation:
    if llm is None:
        return explain_offline(a)
    resp = llm.generate_json(build_messages(a, semantic or SemanticLayer.empty(), audience), EXPLAIN_SCHEMA)
    data = resp.data
    return Explanation(
        analysis=a,
        summary=str(data.get("summary", "")).strip(),
        steps=[str(s).strip() for s in data.get("steps", []) if str(s).strip()],
        output_columns=ground_columns(a, data.get("output_columns", []) or []),
        caveats=[str(c).strip() for c in data.get("caveats", []) if str(c).strip()],
        source="llm",
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


def explain(sql: str, llm: JSONLLMClient | None = None, semantic: SemanticLayer | None = None,
            dialect: str = "duckdb", audience: str = "business") -> list[Explanation]:
    """Explain every statement in ``sql``. Raises ``SQLAnalysisError`` for unparseable input."""
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {sorted(AUDIENCES)}")
    return [explain_analysis(a, llm, semantic, audience) for a in analyze(sql, dialect)]
