"""Evaluation: run a question set through the pipeline and score execution accuracy.

A prediction is correct when its result *data* matches the gold query's result.
Matching is column-based: every gold column must appear among the predicted
columns with the same values (as a multiset, or in order when ``ordered: true``).
Column names, column order and extra columns are ignored, because "total_revenue"
vs "revenue" is not a wrong answer. Numbers are compared at 2-decimal precision.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import sqlglot
import yaml

from .database import Database, QueryResult
from .pipeline import NL2SQL, Answer


@dataclass
class EvalCase:
    id: str
    question: str
    gold_sql: str = ""
    gold_sql_sqlite: str = ""
    category: str = "general"
    difficulty: str = "medium"
    ordered: bool = False
    unanswerable: bool = False

    def gold_for(self, dialect: str) -> str:
        if dialect == "sqlite" and self.gold_sql_sqlite:
            return self.gold_sql_sqlite
        if dialect == "duckdb":
            return self.gold_sql
        return sqlglot.transpile(self.gold_sql, read="duckdb", write=dialect)[0]


@dataclass
class CaseResult:
    id: str
    question: str
    difficulty: str
    category: str
    correct: bool
    reason: str
    predicted_sql: str
    attempts: int
    seconds: float
    input_tokens: int
    output_tokens: int


@dataclass
class EvalReport:
    total: int
    correct: int
    accuracy: float
    by_difficulty: dict[str, dict[str, Any]]
    avg_attempts: float
    first_try_valid_rate: float
    avg_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    results: list[CaseResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text())
    return [EvalCase(**item) for item in raw["questions"]]


def _norm(value: Any) -> Any:
    """Normalise a value so equivalent answers compare equal.

    Numbers -> float rounded to 2 decimals (also numeric strings like '2024');
    dates/midnight timestamps -> 'YYYY-MM-DD'; 'YYYY-MM' -> 'YYYY-MM-01'.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float, Decimal)):
        return round(float(value), 2)
    if isinstance(value, datetime):
        return value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return round(float(text), 2)
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return text + "-01"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} 00:00:00", text):
        return text[:10]
    return text


def _column(result: QueryResult, i: int, ordered: bool) -> list:
    values = [_norm(row[i]) for row in result.rows]
    return values if ordered else sorted(values, key=repr)


def results_match(gold: QueryResult, pred: QueryResult, ordered: bool = False) -> tuple[bool, str]:
    if len(gold.rows) != len(pred.rows):
        return False, f"row count {len(pred.rows)} != expected {len(gold.rows)}"
    if not gold.rows:
        return True, "both empty"
    pred_cols = [_column(pred, i, ordered) for i in range(len(pred.columns))]
    used: set[int] = set()
    for gi, name in enumerate(gold.columns):
        want = _column(gold, gi, ordered)
        hit = next((pi for pi, col in enumerate(pred_cols) if pi not in used and col == want), None)
        if hit is None:
            return False, f"no predicted column matches expected column '{name}'"
        used.add(hit)
    return True, "match"


def score_case(case: EvalCase, answer: Answer, db: Database) -> tuple[bool, str]:
    if case.unanswerable:
        if not answer.sql:
            return True, "correctly declined"
        return False, "answered a question the data cannot answer"
    if not answer.ok:
        return False, answer.error or "no answer"
    gold = db.execute(case.gold_for(db.dialect))
    return results_match(gold, QueryResult(columns=answer.columns, rows=answer.rows), ordered=case.ordered)


def run_eval(pipeline: NL2SQL, cases: list[EvalCase]) -> EvalReport:
    results: list[CaseResult] = []
    for case in cases:
        answer = pipeline.ask(case.question)
        correct, reason = score_case(case, answer, pipeline.db)
        results.append(CaseResult(
            id=case.id, question=case.question, difficulty=case.difficulty, category=case.category,
            correct=correct, reason=reason, predicted_sql=answer.sql, attempts=len(answer.attempts),
            seconds=answer.seconds, input_tokens=answer.input_tokens, output_tokens=answer.output_tokens,
        ))

    n = len(results) or 1
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[r.difficulty].append(r)
    by_difficulty = {
        k: {"total": len(v), "correct": sum(r.correct for r in v),
            "accuracy": round(sum(r.correct for r in v) / len(v), 3)}
        for k, v in groups.items()
    }
    return EvalReport(
        total=len(results),
        correct=sum(r.correct for r in results),
        accuracy=round(sum(r.correct for r in results) / n, 3),
        by_difficulty=by_difficulty,
        avg_attempts=round(sum(r.attempts for r in results) / n, 2),
        first_try_valid_rate=round(sum(r.attempts == 1 for r in results) / n, 3),
        avg_seconds=round(sum(r.seconds for r in results) / n, 3),
        total_input_tokens=sum(r.input_tokens for r in results),
        total_output_tokens=sum(r.output_tokens for r in results),
        results=results,
    )
