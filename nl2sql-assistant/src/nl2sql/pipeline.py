"""The end-to-end pipeline: question -> schema context -> SQL -> validation
(with self-correction) -> results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .database import Database
from .llm import LLMClient
from .prompts import RETRY_PROMPT, build_messages
from .schema_context import build_context
from .semantic import SemanticLayer
from .validator import ValidationError, validate


@dataclass
class Attempt:
    sql: str
    error: str | None = None
    stage: str | None = None


@dataclass
class Answer:
    question: str
    ok: bool
    sql: str = ""
    explanation: str = ""
    assumptions: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    tables_in_context: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    limit_applied: bool = False
    error: str | None = None
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class NL2SQL:
    def __init__(self, db: Database, llm: LLMClient, semantic: SemanticLayer | None = None,
                 max_rows: int = 200, max_attempts: int = 3, max_tables: int | None = 6):
        self.db = db
        self.llm = llm
        self.semantic = semantic or SemanticLayer.empty()
        self.max_rows = max_rows
        self.max_attempts = max_attempts
        self.max_tables = max_tables

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        context = build_context(self.db, self.semantic, question, max_tables=self.max_tables)
        answer = Answer(
            question=question,
            ok=False,
            tables_in_context=context.tables,
            matched_terms=[t.name for t in context.matched_terms],
        )
        messages = build_messages(question, context)

        for _ in range(self.max_attempts):
            resp = self.llm.generate(messages)
            answer.input_tokens += resp.input_tokens
            answer.output_tokens += resp.output_tokens
            answer.explanation, answer.assumptions = resp.explanation, resp.assumptions

            if not resp.sql:  # the model says the question can't be answered
                answer.attempts.append(Attempt(sql="", error="Model declined: " + resp.explanation, stage="unanswerable"))
                answer.error = resp.explanation or "The question can't be answered from this data."
                break
            try:
                checked = validate(resp.sql, self.db, max_rows=self.max_rows)
            except ValidationError as e:
                answer.attempts.append(Attempt(sql=resp.sql, error=e.message, stage=e.stage))
                messages = messages + [
                    {"role": "assistant", "content": resp.sql},
                    {"role": "user", "content": RETRY_PROMPT.format(sql=resp.sql, error=e.message)},
                ]
                continue

            answer.attempts.append(Attempt(sql=resp.sql))
            answer.sql, answer.limit_applied = checked.sql, checked.limit_applied
            try:
                result = self.db.execute(checked.sql)
            except Exception as e:  # noqa: BLE001
                answer.error = f"Query failed at runtime: {e}"
                break
            answer.columns, answer.rows, answer.ok = result.columns, result.rows, True
            break
        else:
            last = answer.attempts[-1]
            answer.error = f"No valid query after {self.max_attempts} attempts. Last error: {last.error}"

        answer.seconds = round(time.perf_counter() - start, 3)
        return answer
