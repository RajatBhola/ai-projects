"""Prompt templates."""

from __future__ import annotations

from .schema_context import SchemaContext

DIALECT_NOTES = {
    "duckdb": (
        "Use DuckDB SQL. Dates: date_trunc('month', d), extract(year FROM d), "
        "strftime(d, '%Y-%m'), d - INTERVAL 30 DAY, date literals as DATE '2025-01-31'."
    ),
    "sqlite": (
        "Use SQLite SQL. Dates are stored as 'YYYY-MM-DD' text: use strftime('%Y', d), "
        "strftime('%Y-%m', d), date(d, '-30 days'); there is no date_trunc or extract."
    ),
}

SYSTEM_PROMPT = """You are a careful analytics engineer who turns business questions into a single SQL query.

Rules:
- Write exactly ONE read-only SELECT query (CTEs allowed). Never modify data.
- Use only the tables and columns listed in the schema. Never invent names.
- When the question uses a term listed under BUSINESS DEFINITIONS, follow that definition exactly.
- Use explicit JOINs with the listed join conditions. Qualify columns with table aliases when joining.
- Give result columns readable aliases (e.g. total_revenue, order_count).
- Round money and percentages to 2 decimals with ROUND(..., 2).
- Order results meaningfully (e.g. largest first for rankings, chronological for time series).
- Do not select personal data (emails) unless the question asks for it.
- If the question is ambiguous, pick the most reasonable interpretation and state it in "assumptions".
- If it cannot be answered from this schema, set "sql" to "" and explain why in "explanation".
{dialect_notes}

Respond with JSON only."""

USER_PROMPT = """SCHEMA
{schema}

QUESTION
{question}"""

RETRY_PROMPT = """Your previous query failed validation.

Previous query:
{sql}

Error:
{error}

Return a corrected query for the same question, as JSON in the same format."""

RESPONSE_SCHEMA = {
    "name": "sql_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "description": "The SQL query, or empty if unanswerable."},
            "explanation": {"type": "string", "description": "One or two sentences on how the query answers the question."},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["sql", "explanation", "assumptions"],
        "additionalProperties": False,
    },
}


def build_messages(question: str, context: SchemaContext) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(dialect_notes=DIALECT_NOTES[context.dialect])},
        {"role": "user", "content": USER_PROMPT.format(schema=context.text, question=question.strip())},
    ]
