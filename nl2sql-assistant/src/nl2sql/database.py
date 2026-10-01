"""Thin database layer over DuckDB or SQLite.

The backend is picked from the file extension: ``.duckdb`` uses DuckDB, anything
else (``.sqlite``, ``.db``) uses Python's built-in sqlite3. Connections used for
answering questions are opened read-only, so even a query that slipped past the
validator cannot change data.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]


class Database:
    def __init__(self, path: str | Path, read_only: bool = True):
        self.path = Path(path)
        self.read_only = read_only
        self.dialect = "duckdb" if self.path.suffix == ".duckdb" else "sqlite"
        self._conn = self._connect()

    def _connect(self):
        if self.dialect == "duckdb":
            import duckdb  # imported lazily so SQLite-only use needs no install

            if self.read_only:
                if not self.path.exists():
                    raise FileNotFoundError(f"Database not found: {self.path} (run `nl2sql seed` first)")
                # enable_external_access=False blocks reading local/remote files
                # through SQL (read_csv, read_parquet, ...). It must be set at connect time.
                return duckdb.connect(
                    str(self.path), read_only=True, config={"enable_external_access": False}
                )
            return duckdb.connect(str(self.path))
        if self.read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"Database not found: {self.path} (run `nl2sql seed` first)")
            return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        return sqlite3.connect(str(self.path), check_same_thread=False)

    # -- querying ---------------------------------------------------------------
    def execute(self, sql: str) -> QueryResult:
        cur = self._conn.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        return QueryResult(columns=columns, rows=[tuple(r) for r in cur.fetchall()])

    def explain(self, sql: str) -> None:
        """Ask the engine to plan the query without running it. Raises on errors."""
        prefix = "EXPLAIN QUERY PLAN " if self.dialect == "sqlite" else "EXPLAIN "
        self._conn.execute(prefix + sql).fetchall()

    def executescript(self, statements: list[str]) -> None:
        for stmt in statements:
            self._conn.execute(stmt)
        if self.dialect == "sqlite":
            self._conn.commit()

    def insert_many(self, table: str, rows: list[tuple], batch_size: int = 150) -> None:
        """Insert rows with multi-row VALUES statements (much faster than
        executemany on DuckDB; 150 rows keeps SQLite under its variable limit)."""
        if not rows:
            return
        row_marks = "(" + ", ".join(["?"] * len(rows[0])) + ")"
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            sql = f"INSERT INTO {table} VALUES " + ", ".join([row_marks] * len(batch))
            self._conn.execute(sql, [v for row in batch for v in row])
        if self.dialect == "sqlite":
            self._conn.commit()

    # -- introspection ----------------------------------------------------------
    def tables(self) -> dict[str, list[Column]]:
        """Return {table_name: [Column, ...]} for every user table."""
        if self.dialect == "sqlite":
            names = [
                r[0]
                for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            return {
                name: [Column(r[1], r[2]) for r in self._conn.execute(f"PRAGMA table_info('{name}')").fetchall()]
                for name in names
            }
        rows = self._conn.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ).fetchall()
        out: dict[str, list[Column]] = {}
        for table, column, dtype in rows:
            out.setdefault(table, []).append(Column(column, dtype))
        return out

    def distinct_values(self, table: str, column: str, limit: int = 12) -> list[Any] | None:
        """Distinct values of a column, or None if it has more than ``limit``."""
        rows = self._conn.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {limit + 1}'
        ).fetchall()
        if len(rows) > limit:
            return None
        return sorted(r[0] for r in rows)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
