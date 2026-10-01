from datetime import date
from decimal import Decimal

from nl2sql.database import QueryResult
from nl2sql.evaluation import load_cases, results_match, run_eval
from nl2sql.llm import ScriptedLLM
from nl2sql.pipeline import NL2SQL


def qr(columns, rows):
    return QueryResult(columns=columns, rows=rows)


def test_match_ignores_column_names_order_and_extras():
    gold = qr(["country", "revenue"], [("NL", 10.0), ("DE", 5.0)])
    pred = qr(["total_rev", "cnt", "c"], [(5.0, 1, "DE"), (10.0, 2, "NL")])
    assert results_match(gold, pred)[0]


def test_match_normalises_numbers_and_dates():
    gold = qr(["year", "month", "amount"], [("2024", "2024-01", Decimal("10.004"))])
    pred = qr(["y", "m", "a"], [(2024, date(2024, 1, 1), 10.0)])
    assert results_match(gold, pred)[0]


def test_match_detects_wrong_values_and_row_counts():
    gold = qr(["n"], [(1,), (2,)])
    assert not results_match(gold, qr(["n"], [(1,), (3,)]))[0]
    assert not results_match(gold, qr(["n"], [(1,)]))[0]


def test_ordered_comparison():
    gold = qr(["m"], [("a",), ("b",)])
    assert results_match(gold, qr(["m"], [("b",), ("a",)]), ordered=False)[0]
    assert not results_match(gold, qr(["m"], [("b",), ("a",)]), ordered=True)[0]


def test_every_gold_query_runs(db, questions_path):
    for case in load_cases(questions_path):
        if case.unanswerable:
            continue
        result = db.execute(case.gold_for(db.dialect))
        assert result.rows, f"{case.id} returned no rows"


def test_perfect_model_scores_100_percent(db, semantic, questions_path):
    cases = load_cases(questions_path)
    llm = ScriptedLLM([c.gold_for(db.dialect) if not c.unanswerable else "" for c in cases])
    report = run_eval(NL2SQL(db, llm, semantic), cases)
    failures = [(r.id, r.reason) for r in report.results if not r.correct]
    assert report.accuracy == 1.0, failures


def test_wrong_model_is_penalised(db, semantic, questions_path):
    cases = [c for c in load_cases(questions_path) if c.id in {"e01", "u01"}]
    llm = ScriptedLLM(["SELECT COUNT(*) FROM orders", "SELECT 1"])  # wrong count, answers unanswerable
    report = run_eval(NL2SQL(db, llm, semantic), cases)
    assert report.correct == 0
