from nl2sql.seed import generate


def test_generation_is_deterministic():
    a, b = generate(seed=7), generate(seed=7)
    assert a == b
    assert generate(seed=8)["orders"] != a["orders"]


def test_expected_sizes(db):
    counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").rows[0][0]
              for t in ("customers", "products", "orders", "order_items")}
    assert counts["customers"] == 600
    assert counts["products"] == 40
    assert counts["orders"] == 6000
    assert counts["order_items"] > counts["orders"]


def test_some_customers_never_ordered(db):
    n = db.execute(
        "SELECT COUNT(*) FROM customers c WHERE NOT EXISTS "
        "(SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id)"
    ).rows[0][0]
    assert 0 < n < 100


def test_orders_never_precede_signup(db):
    n = db.execute(
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE o.order_date < c.signup_date"
    ).rows[0][0]
    assert n == 0
