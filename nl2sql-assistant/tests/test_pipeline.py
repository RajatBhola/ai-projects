import sys
from types import SimpleNamespace

import pytest

from nl2sql.llm import ScriptedLLM, parse_response
from nl2sql.pipeline import NL2SQL


def test_answers_on_first_try(db, semantic):
    llm = ScriptedLLM(["SELECT COUNT(*) AS n FROM customers"])
    answer = NL2SQL(db, llm, semantic).ask("How many customers?")
    assert answer.ok
    assert answer.rows == [(600,)]
    assert len(answer.attempts) == 1


def test_self_corrects_after_validation_error(db, semantic):
    llm = ScriptedLLM([
        "SELECT SUM(revenue) FROM orders",  # hallucinated column
        "SELECT COUNT(*) AS n FROM orders",
    ])
    answer = NL2SQL(db, llm, semantic).ask("How many orders?")
    assert answer.ok
    assert [a.stage for a in answer.attempts] == ["schema", None]
    retry_prompt = llm.calls[1][-1]["content"]
    assert "failed validation" in retry_prompt and "revenue" in retry_prompt


def test_gives_up_after_max_attempts(db, semantic):
    llm = ScriptedLLM(["DROP TABLE orders"] * 5)
    answer = NL2SQL(db, llm, semantic, max_attempts=3).ask("Delete everything")
    assert not answer.ok
    assert len(answer.attempts) == 3
    assert "No valid query after 3 attempts" in answer.error


def test_unanswerable_question_is_declined(db, semantic):
    llm = ScriptedLLM([""], explanation="There is no cost data, so profit cannot be computed.")
    answer = NL2SQL(db, llm, semantic).ask("What is our profit margin?")
    assert not answer.ok
    assert answer.sql == ""
    assert "cost data" in answer.error


def test_prompt_contains_schema_and_dialect(db, semantic):
    llm = ScriptedLLM(["SELECT 1 AS x"])
    NL2SQL(db, llm, semantic).ask("revenue by country")
    system, user = llm.calls[0][0]["content"], llm.calls[0][1]["content"]
    assert ("DuckDB" if db.dialect == "duckdb" else "SQLite") in system
    assert "TABLE orders" in user and "revenue by country" in user


@pytest.mark.parametrize("text", [
    '{"sql": "SELECT 1", "explanation": "x", "assumptions": []}',
    '```json\n{"sql": "SELECT 1", "explanation": "x", "assumptions": []}\n```',
])
def test_parse_response(text):
    assert parse_response(text).sql == "SELECT 1"


def test_openai_client_requests_structured_output(monkeypatch):
    """Exercise OpenAIClient against a fake SDK: no network or API key needed."""
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content='{"sql": "SELECT 1", "explanation": "ok", "assumptions": ["a"]}')
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                                   usage=SimpleNamespace(prompt_tokens=120, completion_tokens=15))

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    from nl2sql.llm import OpenAIClient

    resp = OpenAIClient(model="test-model").generate([{"role": "user", "content": "hi"}])
    assert (resp.sql, resp.assumptions, resp.input_tokens, resp.output_tokens) == ("SELECT 1", ["a"], 120, 15)
    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.0
    assert captured["response_format"]["type"] == "json_schema"
