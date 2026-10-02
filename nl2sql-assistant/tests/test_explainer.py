import io
import json
import sys

import pytest

from nl2sql.cli import main
from nl2sql.explainer import EXPLAIN_SCHEMA, explain
from nl2sql.llm import ScriptedJSONLLM

SQL = ("SELECT c.country, SUM(oi.quantity * oi.unit_price) AS revenue "
       "FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
       "JOIN order_items oi ON oi.order_id = o.order_id "
       "WHERE o.status = 'completed' GROUP BY c.country ORDER BY revenue DESC LIMIT 3")

MODEL_ANSWER = {
    "summary": "The 3 countries with the most revenue from completed orders.",
    "steps": ["Takes completed orders.", "Adds up spend per country.", "Keeps the top 3."],
    "output_columns": [
        {"name": "Revenue", "meaning": "Total spent in EUR."},   # different case: still matched
        {"name": "country", "meaning": "Customer's country."},
        {"name": "made_up", "meaning": "Not in the query."},    # must be dropped
    ],
    "caveats": ["Discounts are not subtracted, unlike the official revenue definition."],
}


def test_llm_explanation_is_grounded(semantic):
    llm = ScriptedJSONLLM([MODEL_ANSWER])
    [ex] = explain(SQL, llm=llm, semantic=semantic)
    assert ex.source == "llm"
    assert ex.summary.startswith("The 3 countries")
    assert [(c.name, c.meaning) for c in ex.output_columns] == [
        ("country", "Customer's country."), ("revenue", "Total spent in EUR."),
    ]
    assert ex.caveats and (ex.input_tokens, ex.output_tokens) == (100, 50)


def test_prompt_contains_parser_facts_and_meanings(semantic):
    llm = ScriptedJSONLLM([MODEL_ANSWER])
    explain(SQL, llm=llm, semantic=semantic, audience="technical")
    messages, schema = llm.calls[0]
    system, user = messages[0]["content"], messages[1]["content"]
    assert schema is EXPLAIN_SCHEMA
    assert "analyst or engineer" in system
    assert '"kind": "INNER"' in user and "o.status = 'completed'" in user          # parser facts
    assert "Only completed orders count as sales" in user                        # column meaning
    assert "revenue: Net revenue in EUR" in user                                 # business definition
    assert "products" not in user.split("TABLE DESCRIPTIONS")[1].split("BUSINESS")[0]  # only used tables


def test_detected_issues_reach_the_prompt(semantic):
    llm = ScriptedJSONLLM([MODEL_ANSWER])
    [ex] = explain("DELETE FROM orders", llm=llm, semantic=semantic)
    user = llm.calls[0][0][1]["content"]
    assert "affects every row in orders" in user
    assert {i.code for i in ex.analysis.issues} >= {"modifies_data", "no_where"}


def test_offline_needs_no_llm():
    [ex] = explain(SQL)
    assert ex.source == "offline"
    assert "read-only" in ex.summary
    assert ex.steps[-1] == "Returns at most 3 rows."


def test_one_explanation_per_statement():
    assert len(explain("SELECT 1 AS a; SELECT 2 AS b")) == 2


def test_unknown_audience():
    with pytest.raises(ValueError):
        explain(SQL, audience="kids")


# ----------------------------------------------------------------------------- CLI
def run(capsys, *argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_offline(capsys):
    code, out, _ = run(capsys, "explain", "--offline", SQL)
    assert code == 0
    assert "Step by step" in out and "Returns at most 3 rows." in out
    assert "offline (no LLM)" in out


def test_cli_stdin_json_and_warnings(capsys, monkeypatch):
    code, out, _ = run(capsys, "explain", "--offline", "--json", "-",
                       stdin="UPDATE orders SET status = 'x'", monkeypatch=monkeypatch)
    data = json.loads(out)
    assert code == 0 and data["modifies_data"] is True
    assert {i["code"] for i in data["issues"]} >= {"modifies_data", "no_where"}


def test_cli_file_and_dialect(capsys, tmp_path):
    f = tmp_path / "q.sql"
    f.write_text("SELECT TOP 3 name FROM customers ORDER BY name")
    code, out, _ = run(capsys, "explain", "--offline", "--dialect", "tsql", "--file", str(f))
    assert code == 0 and "Returns at most 3 rows." in out


def test_cli_rejects_bad_input(capsys):
    assert run(capsys, "explain", "--offline", "SELEC * FRM x")[0] == 2
    assert run(capsys, "explain", "--offline", "--dialect", "klingon", "SELECT 1")[0] == 2


def test_cli_with_llm(capsys, monkeypatch):
    fake = ScriptedJSONLLM([MODEL_ANSWER])
    monkeypatch.setattr("nl2sql.llm.OpenAIClient", lambda model: fake)
    code, out, _ = run(capsys, "explain", SQL)
    assert code == 0
    assert "The 3 countries" in out and "Watch out" in out and "made_up" not in out
