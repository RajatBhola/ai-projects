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


def parse_response(text: str) -> LLMResponse:
    """Parse the model's JSON. Tolerates ```json fences in case a model adds them."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
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

    def generate(self, messages: list[dict]) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
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
        result = parse_response(resp.choices[0].message.content or "{}")
        if resp.usage:
            result.input_tokens = resp.usage.prompt_tokens
            result.output_tokens = resp.usage.completion_tokens
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
