"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader: KEY=VALUE lines, existing env vars win."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    db_path: Path
    semantic_layer_path: Path
    openai_model: str
    max_rows: int
    max_attempts: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        db = Path(os.environ.get("NL2SQL_DB", "data/warehouse.duckdb"))
        if not db.is_absolute():
            db = PROJECT_ROOT / db
        semantic = Path(os.environ.get("NL2SQL_SEMANTIC_LAYER", "semantic_layer.yaml"))
        if not semantic.is_absolute():
            semantic = PROJECT_ROOT / semantic
        return cls(
            db_path=db,
            semantic_layer_path=semantic,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            max_rows=int(os.environ.get("NL2SQL_MAX_ROWS", "200")),
            max_attempts=int(os.environ.get("NL2SQL_MAX_ATTEMPTS", "3")),
        )
