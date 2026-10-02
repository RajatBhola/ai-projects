import pytest

from nl2sql.analyzer import SQLAnalysisError, analyze, describe, strip_jinja


def one(sql, dialect="duckdb"):
    [a] = analyze(sql, dialect)
    return a


def codes(a):
    return {i.code for i in a.issues}


REVENUE_SQL = """
SELECT c.country, SUM(oi.quantity * oi.unit_price) AS revenue, COUNT(*)
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status = 'completed' AND o.order_date >= DATE '2025-01-01'
GROUP BY 1
HAVING SUM(oi.quantity) > 10
ORDER BY 2 DESC
LIMIT 5
"""


def test_select_structure():
    a = one(REVENUE_SQL)
    assert a.statement_type == "SELECT" and not a.modifies_data
    assert a.tables == ["orders", "customers", "order_items"]
    assert a.source == "orders AS o"
    assert [(j.kind, j.table) for j in a.joins] == [("INNER", "customers AS c"), ("INNER", "order_items AS oi")]
    assert a.filters[0] == "o.status = 'completed'" and len(a.filters) == 2
    assert a.group_by == ["country"]                  # positional GROUP BY 1 resolved
    assert a.order_by == ["revenue DESC"]             # positional ORDER BY 2 resolved
    assert a.limit == "5"
    assert a.having == ["SUM(oi.quantity) > 10"]
    assert [c.name for c in a.output_columns] == ["country", "revenue", "COUNT(*)"]
    assert "SUM(oi.quantity * oi.unit_price)" in a.aggregations
    assert a.issues == []                             # a clean query raises nothing


def test_ctes_subqueries_windows_and_distinct():
    a = one("""
        WITH big AS (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 5)
        SELECT DISTINCT p.category,
               RANK() OVER (ORDER BY p.list_price DESC) AS price_rank
        FROM products p
        WHERE p.product_id IN (SELECT product_id FROM order_items)
        LIMIT 10
    """)
    assert a.ctes == ["big"]
    assert "big" not in a.tables and "order_items" in a.tables
    assert a.subqueries == 1
    assert a.distinct
    assert a.window_functions and "RANK()" in a.window_functions[0]


def test_union():
    a = one("SELECT a FROM x UNION ALL SELECT a FROM y")
    assert a.statement_type == "UNION ALL" and a.union_branches == 2
    assert "union_distinct" not in codes(a)
    assert "union_distinct" in codes(one("SELECT a FROM x UNION SELECT a FROM y"))


@pytest.mark.parametrize("sql,kind,target", [
    ("DELETE FROM orders WHERE status = 'cancelled'", "DELETE", "orders"),
    ("UPDATE orders SET status = 'x' WHERE order_id = 1", "UPDATE", "orders"),
    ("INSERT INTO archive SELECT * FROM orders", "INSERT", "archive"),
    ("CREATE TABLE t AS SELECT 1 AS x", "CREATE TABLE", "t"),
    ("DROP TABLE orders", "DROP TABLE", "orders"),
])
def test_writes_are_flagged(sql, kind, target):
    a = one(sql)
    assert a.statement_type == kind and a.modifies_data and a.target_table == target
    assert "modifies_data" in codes(a)


def test_update_details_and_no_where():
    a = one("UPDATE orders SET status = 'x', channel = 'web'")
    assert a.set_columns == ["status", "channel"]
    assert "no_where" in codes(a)
    assert "no_where" not in codes(one("DELETE FROM orders WHERE order_id = 1"))
    assert "no_where" in codes(one("DELETE FROM orders"))


@pytest.mark.parametrize("sql,code", [
    ("SELECT * FROM orders WHERE customer_id = NULL", "null_comparison"),
    ("SELECT * FROM orders WHERE customer_id != NULL", "null_comparison"),
    ("SELECT id FROM a WHERE id NOT IN (SELECT a_id FROM b) LIMIT 5", "not_in_subquery"),
    ("SELECT o.id FROM orders o LEFT JOIN returns r ON r.order_id = o.id WHERE r.reason = 'x' LIMIT 5",
     "left_join_filtered"),
    ("SELECT a.x FROM a, b LIMIT 5", "missing_join_condition"),
    ("SELECT * FROM orders LIMIT 5", "select_star"),
    ("SELECT id FROM orders", "no_limit"),
])
def test_detects_issue(sql, code):
    assert code in codes(one(sql))


@pytest.mark.parametrize("sql,code", [
    # anti-join: filtering the LEFT JOINed table on IS NULL is deliberate
    ("SELECT o.id FROM orders o LEFT JOIN returns r ON r.order_id = o.id WHERE r.order_id IS NULL LIMIT 5",
     "left_join_filtered"),
    # comma join linked in WHERE is a real join
    ("SELECT a.x FROM a, b WHERE a.id = b.id LIMIT 5", "missing_join_condition"),
    ("SELECT a.x FROM a CROSS JOIN b LIMIT 5", "missing_join_condition"),
    ("SELECT id FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.a_id = a.id) LIMIT 5", "not_in_subquery"),
    ("SELECT COUNT(*) FROM orders", "no_limit"),       # aggregates return one row
    ("SELECT * FROM orders WHERE customer_id IS NULL LIMIT 5", "null_comparison"),
])
def test_no_false_positive(sql, code):
    assert code not in codes(one(sql))


def test_multiple_statements():
    results = analyze("SELECT 1 AS x; DELETE FROM orders")
    assert [a.statement_type for a in results] == ["SELECT", "DELETE"]


@pytest.mark.parametrize("sql", ["SELEC * FRM orders", "hello world", "   "])
def test_rejects_non_sql(sql):
    with pytest.raises(SQLAnalysisError):
        analyze(sql)


def test_dialects():
    a = one("SELECT TOP 5 name FROM [dbo].[customers] ORDER BY name", dialect="tsql")
    assert a.limit == "5" and a.tables == ["dbo.customers"]
    a = one("SELECT * FROM `project.dataset.orders` QUALIFY ROW_NUMBER() OVER (PARTITION BY id) = 1",
            dialect="bigquery")
    assert a.tables and a.tables[0].endswith("orders")


def test_dbt_jinja():
    sql = ("{{ config(materialized='table') }}\n"
           "select o.id from {{ ref('stg_orders') }} o "
           "join {{ source('shop', 'customers') }} c on c.id = o.customer_id "
           "{% if is_incremental() %} where o.updated_at > {{ var('since') }} {% endif %}")
    cleaned, changed = strip_jinja(sql)
    assert changed and "{{" not in cleaned and "{%" not in cleaned
    a = one(sql)
    assert a.tables == ["stg_orders", "shop.customers"]
    assert "jinja" in codes(a)
    assert strip_jinja("select 1")[1] is False


def test_describe_reads_naturally():
    steps = describe(one(REVENUE_SQL))
    text = " ".join(steps)
    assert steps[0] == "Reads from orders AS o."
    assert "keeping only rows that match on both sides" in text
    assert "Groups the rows by country" in text
    assert "Sorts by revenue (highest first)." in text
    assert steps[-1] == "Returns at most 5 rows."
    assert describe(one("DELETE FROM orders")) == ["Deletes rows from orders (every row)."]
