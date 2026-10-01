from nl2sql.schema_context import build_context
from nl2sql.semantic import mentions


def test_mentions_whole_words_and_plurals():
    assert mentions("How many clients?", "client")
    assert mentions("total revenue in 2025", "revenue")
    assert mentions("What is the AOV?", "aov")
    assert not mentions("total revenue", "rev")
    assert not mentions("reorders", "orders")


def test_matches_business_terms_via_synonyms(semantic):
    names = {t.name for t in semantic.match_terms("What was our turnover by country?")}
    assert names == {"revenue"}
    assert {t.name for t in semantic.match_terms("What's the AOV per segment?")} == {"average order value"}
    assert semantic.match_terms("How many products are there?") == []


def test_context_includes_definitions_values_and_joins(db, semantic):
    ctx = build_context(db, semantic, "What was total revenue by country in 2025?")
    assert "BUSINESS DEFINITIONS" in ctx.text
    assert "orders.status = 'completed'" in ctx.text
    assert "[values: 'cancelled', 'completed', 'returned']" in ctx.text
    assert "order_items.order_id = orders.order_id" in ctx.text
    assert ctx.dialect == db.dialect


def test_context_hides_values_of_personal_columns(db, semantic):
    ctx = build_context(db, semantic, "list customers")
    email_line = next(line for line in ctx.text.splitlines() if line.strip().startswith("email"))
    assert "values:" not in email_line


def test_table_selection_adds_join_path(db, semantic):
    # "category" lives in products and "revenue" needs orders + order_items:
    # with a budget of 2, the join path to connect them must still be included.
    ctx = build_context(db, semantic, "Which product category has the most revenue?", max_tables=2)
    assert {"products", "order_items", "orders"} <= set(ctx.tables)


def test_table_selection_keeps_single_relevant_table(db, semantic):
    ctx = build_context(db, semantic, "How many clients signed up?", max_tables=2)
    assert ctx.tables[0] == "customers"


def test_business_term_pulls_in_its_tables(db, semantic):
    ctx = build_context(db, semantic, "What is the AOV per customer segment?", max_tables=2)
    assert {"orders", "order_items", "customers"} <= set(ctx.tables)
