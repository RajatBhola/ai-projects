"""Load the semantic layer and match a question's wording to it."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TableInfo:
    name: str
    description: str = ""
    synonyms: list[str] = field(default_factory=list)
    columns: dict[str, str] = field(default_factory=dict)


@dataclass
class BusinessTerm:
    name: str
    definition: str
    synonyms: list[str] = field(default_factory=list)

    @property
    def phrases(self) -> list[str]:
        return [self.name, *self.synonyms]


@dataclass
class SemanticLayer:
    tables: dict[str, TableInfo]
    relationships: list[str]
    business_terms: dict[str, BusinessTerm]

    @classmethod
    def load(cls, path: str | Path) -> "SemanticLayer":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def empty(cls) -> "SemanticLayer":
        return cls(tables={}, relationships=[], business_terms={})

    @classmethod
    def from_dict(cls, raw: dict) -> "SemanticLayer":
        tables = {
            name: TableInfo(
                name=name,
                description=(spec or {}).get("description", ""),
                synonyms=list((spec or {}).get("synonyms", [])),
                columns={k: str(v) for k, v in ((spec or {}).get("columns") or {}).items()},
            )
            for name, spec in (raw.get("tables") or {}).items()
        }
        terms = {
            name: BusinessTerm(
                name=name,
                definition=" ".join(str(spec.get("definition", "")).split()),
                synonyms=list(spec.get("synonyms", [])),
            )
            for name, spec in (raw.get("business_terms") or {}).items()
        }
        return cls(tables=tables, relationships=list(raw.get("relationships") or []), business_terms=terms)

    def match_terms(self, question: str) -> list[BusinessTerm]:
        """Business terms whose name or a synonym appears in the question."""
        return [t for t in self.business_terms.values() if any(mentions(question, p) for p in t.phrases)]

    def joins(self) -> list[tuple[str, str, str]]:
        """Relationships as (left_table, right_table, condition) triples."""
        out = []
        for rel in self.relationships:
            left, right = (side.strip() for side in rel.split("="))
            out.append((left.split(".")[0], right.split(".")[0], rel))
        return out


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def mentions(question: str, phrase: str) -> bool:
    """Whole-word, case-insensitive match that tolerates a plural 's'.

    ``mentions("How many clients?", "client")`` -> True
    ``mentions("total revenue", "rev")``        -> False
    """
    q = f" {_normalise(question)} "
    p = _normalise(phrase)
    if not p:
        return False
    return f" {p} " in q or f" {p}s " in q or (p.endswith("s") and f" {p[:-1]} " in q)
