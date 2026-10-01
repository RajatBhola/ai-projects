"""Query validation: every generated query passes these checks before it runs.

Layers, cheapest first:
1. Parse        - must be valid SQL in the target dialect, exactly one statement.
2. Read-only    - must be a SELECT/WITH/UNION query; no DML, DDL, PRAGMA, ATTACH...
3. Safe functions - no file/network access functions (read_csv, load_extension...).
4. Schema check - every table and column must exist (catches hallucinated names).
5. Row limit    - a LIMIT is added if missing, and capped if too large.
6. Dry run      - the database plans the query (EXPLAIN) without executing it.

Each failure raises ``ValidationError`` with a message written for the LLM, so the
pipeline can feed it back and ask for a corrected query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

from .database import Database

BLOCKED_FUNCTIONS = {
    # DuckDB file / network access
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
    "read_json_auto", "read_json_objects", "read_ndjson", "read_text", "read_blob",
    "glob", "sniff_csv", "iceberg_scan", "delta_scan", "sqlite_scan", "postgres_scan",
    "mysql_scan", "getenv", "query", "query_table",
    # SQLite extensions / file access
    "load_extension", "readfile", "writefile", "fts3_tokenizer",
}

FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
    exp.Command, exp.Pragma, exp.Attach, exp.Detach, exp.Copy, exp.Set, exp.Use,
    exp.Transaction, exp.Commit, exp.Rollback, exp.TruncateTable, exp.Into,
)


class ValidationError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


@dataclass
class ValidatedQuery:
    sql: str                      # the query that will actually run
    original_sql: str             # what the model produced
    tables: list[str] = field(default_factory=list)
    limit_applied: bool = False


def _forbidden_node_types() -> tuple[type, ...]:
    # Some node classes only exist in some sqlglot versions.
    return tuple(t for t in FORBIDDEN_NODES if isinstance(t, type))


def _function_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.this).lower()
    return (node.sql_name() or "").lower()


def validate(sql: str, db: Database, max_rows: int = 200, dry_run: bool = True) -> ValidatedQuery:
    dialect = db.dialect
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValidationError("parse", "The query is empty.")

    # 1. Parse
    try:
        statements = [s for s in sqlglot.parse(cleaned, read=dialect) if s is not None]
    except ParseError as e:
        raise ValidationError("parse", f"SQL syntax error: {e.errors[0]['description'] if e.errors else e}") from e
    if len(statements) != 1:
        raise ValidationError("parse", f"Expected exactly one statement, got {len(statements)}.")
    tree = statements[0]

    # 2. Read-only
    if not isinstance(tree, exp.Query):
        raise ValidationError("read_only", f"Only SELECT queries are allowed, got {tree.key.upper()}.")
    bad = next(iter(tree.find_all(*_forbidden_node_types())), None)
    if bad is not None:
        raise ValidationError("read_only", f"Statement type {bad.key.upper()} is not allowed; only read data.")

    # 3. Safe functions (table functions such as read_csv('...') included)
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name in BLOCKED_FUNCTIONS:
            raise ValidationError("unsafe_function", f"Function {name}() is not allowed.")

    # 4. Schema check
    schema = db.tables()
    known = {t.lower() for t in schema}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    referenced = []
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name or name in cte_names:
            continue
        if name not in known:
            raise ValidationError(
                "schema", f"Unknown table '{table.name}'. Available tables: {', '.join(sorted(schema))}."
            )
        referenced.append(name)

    mapping = {t: {c.name: c.type for c in cols} for t, cols in schema.items()}
    try:
        qualify(tree.copy(), schema=mapping, dialect=dialect, validate_qualify_columns=True,
                identify=False, quote_identifiers=False)
    except OptimizeError as e:
        raise ValidationError("schema", f"{e}. Check column names against the schema.") from e

    # 5. Row limit
    limit_applied = False
    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
        limit_applied = True
    else:
        try:
            current = int(limit.expression.name) if isinstance(limit, exp.Limit) else max_rows
        except (TypeError, ValueError):
            current = max_rows + 1
        if current > max_rows:
            tree = tree.limit(max_rows)
            limit_applied = True
    final_sql = tree.sql(dialect=dialect, pretty=True)

    # 6. Dry run
    if dry_run:
        try:
            db.explain(final_sql)
        except Exception as e:  # noqa: BLE001 - database errors vary by backend
            raise ValidationError("dry_run", f"The database rejected the query: {e}") from e

    return ValidatedQuery(sql=final_sql, original_sql=sql, tables=sorted(set(referenced)),
                          limit_applied=limit_applied)
