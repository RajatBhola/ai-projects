import pytest

from nl2sql.database import Column
from nl2sql.validator import ValidationError, validate

GOOD = [
    "SELECT country, COUNT(*) AS n FROM customers GROUP BY country",
    "WITH c AS (SELECT customer_id FROM orders) SELECT COUNT(*) FROM c",
    "SELECT o.order_id, c.full_name FROM orders o JOIN customers c ON o.customer_id = c.customer_id",
    "SELECT status FROM orders UNION SELECT segment FROM customers",
    "SELECT status, COUNT(*) AS n FROM orders GROUP BY status HAVING COUNT(*) > 10",
    "SELECT * FROM orders WHERE customer_id IN (SELECT customer_id FROM customers WHERE country = 'Spain')",
]

BAD = [
    ("DELETE FROM orders", "read_only"),
    ("DROP TABLE orders", "read_only"),
    ("UPDATE orders SET status = 'x'", "read_only"),
    ("INSERT INTO orders SELECT * FROM orders", "read_only"),
    ("SELECT 1; DROP TABLE orders", "parse"),
    ("SELEC * FROM orders", "parse"),
    ("", "parse"),
    ("SELECT * FROM revenue", "schema"),
    ("SELECT revenue FROM orders", "schema"),
    ("SELECT o.revenue FROM orders o", "schema"),
    ("SELECT * FROM read_csv('/etc/passwd')", "unsafe_function"),
    ("SELECT load_extension('evil')", "unsafe_function"),
]


@pytest.mark.parametrize("sql", GOOD)
def test_accepts_valid_queries(sql, db):
    result = validate(sql, db)
    db.execute(result.sql)  # and it really runs


@pytest.mark.parametrize("sql,stage", BAD)
def test_rejects_unsafe_or_invalid_queries(sql, stage, db):
    with pytest.raises(ValidationError) as exc:
        validate(sql, db)
    assert exc.value.stage == stage


def test_adds_limit_when_missing(db):
    result = validate("SELECT * FROM orders", db, max_rows=10)
    assert result.limit_applied
    assert len(db.execute(result.sql).rows) == 10


def test_caps_large_limit(db):
    result = validate("SELECT * FROM orders LIMIT 100000", db, max_rows=50)
    assert result.limit_applied
    assert len(db.execute(result.sql).rows) == 50


def test_keeps_small_limit(db):
    result = validate("SELECT * FROM orders LIMIT 5", db, max_rows=50)
    assert not result.limit_applied
    assert len(db.execute(result.sql).rows) == 5


def test_reports_referenced_tables(db):
    result = validate("SELECT * FROM orders o JOIN customers c ON o.customer_id = c.customer_id", db)
    assert result.tables == ["customers", "orders"]


def test_error_message_lists_available_tables(db):
    with pytest.raises(ValidationError, match="Available tables: customers, order_items, orders, products"):
        validate("SELECT * FROM sales", db)


class FakeDuckDB:
    """Parse-level checks in the DuckDB dialect without needing duckdb installed."""

    dialect = "duckdb"

    def tables(self):
        return {"orders": [Column("order_id", "INTEGER"), Column("status", "VARCHAR")]}


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM read_parquet('s3://bucket/x.parquet')",
    "SELECT * FROM read_json_auto('x.json')",
    "SELECT * FROM glob('/home/*')",
    "SELECT getenv('OPENAI_API_KEY')",
])
def test_blocks_duckdb_file_access(sql):
    with pytest.raises(ValidationError) as exc:
        validate(sql, FakeDuckDB(), dry_run=False)
    assert exc.value.stage in {"unsafe_function", "schema"}


@pytest.mark.parametrize("sql", ["ATTACH 'other.db' AS other", "COPY orders TO 'out.csv'", "PRAGMA version"])
def test_blocks_duckdb_admin_statements(sql):
    with pytest.raises(ValidationError):
        validate(sql, FakeDuckDB(), dry_run=False)
