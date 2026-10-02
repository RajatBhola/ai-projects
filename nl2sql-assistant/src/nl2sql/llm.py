"""LLM clients. The pipeline depends only on the small ``LLMClient`` protocol, so
tests use a scripted fake and swapping providers means adding one class."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from .prompts import RESPONSE_SCHEMA


@dataclass
class LLMResponse:
    sql: str
    explanation: str
    assumptions: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    def generate(self, messages: list[dict]) -> LLMResponse: ...


@dataclass
class JSONResponse:
    data: dict
    input_tokens: int = 0
    output_tokens: int = 0


class JSONLLMClient(Protocol):
    def generate_json(self, messages: list[dict], json_schema: dict) -> JSONResponse: ...


def _load_json(text: str) -> dict:
    """Parse JSON, tolerating ```json fences in case a model adds them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def parse_response(text: str) -> LLMResponse:
    """Parse the SQL-generation JSON into an LLMResponse."""
    data = _load_json(text)
    return LLMResponse(
        sql=str(data.get("sql", "")).strip(),
        explanation=str(data.get("explanation", "")).strip(),
        assumptions=[str(a) for a in data.get("assumptions", []) or []],
    )


class OpenAIClient:
    """OpenAI Chat Completions with structured (JSON-schema) output."""

    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None, temperature: float | None = 0.0):
        from openai import OpenAI  # lazy import: tests and seeding don't need it

        self.client = OpenAI(api_key=api_key)  # falls back to OPENAI_API_KEY
        self.model = model
        self.temperature = temperature

    def generate_json(self, messages: list[dict], json_schema: dict) -> JSONResponse:
        """Chat completion constrained to ``json_schema``; returns the parsed object."""
        kwargs = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": json_schema},
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            # Some (reasoning) models reject a custom temperature: retry without it.
            if "temperature" not in str(e) or "temperature" not in kwargs:
                raise
            self.temperature = None
            kwargs.pop("temperature")
            resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return JSONResponse(
            data=_load_json(resp.choices[0].message.content or "{}"),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    def generate(self, messages: list[dict]) -> LLMResponse:
        resp = self.generate_json(messages, RESPONSE_SCHEMA)
        result = parse_response(json.dumps(resp.data))
        result.input_tokens, result.output_tokens = resp.input_tokens, resp.output_tokens
        return result


class ScriptedLLM:
    """Test double: returns the given SQL strings in order and records the prompts."""

    def __init__(self, sqls: list[str], explanation: str = "scripted"):
        self.sqls = list(sqls)
        self.explanation = explanation
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> LLMResponse:
        self.calls.append(messages)
        sql = self.sqls.pop(0) if self.sqls else ""
        return LLMResponse(sql=sql, explanation=self.explanation)


class ScriptedJSONLLM:
    """Test double for ``generate_json``: returns the given objects in order."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls: list[tuple[list[dict], dict]] = []

    def generate_json(self, messages: list[dict], json_schema: dict) -> JSONResponse:
        self.calls.append((messages, json_schema))
        return JSONResponse(data=self.responses.pop(0) if self.responses else {},
                            input_tokens=100, output_tokens=50)
