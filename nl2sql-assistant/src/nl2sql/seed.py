"""Generate a small, deterministic e-commerce warehouse to query.

Same seed -> same data, so the evaluation set's expected answers never drift.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from .database import Database

DDL = [
    """CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        full_name   VARCHAR NOT NULL,
        email       VARCHAR NOT NULL,
        country     VARCHAR NOT NULL,
        segment     VARCHAR NOT NULL,
        signup_date DATE NOT NULL
    )""",
    """CREATE TABLE products (
        product_id  INTEGER PRIMARY KEY,
        product_name VARCHAR NOT NULL,
        category    VARCHAR NOT NULL,
        list_price  DECIMAL(10, 2) NOT NULL
    )""",
    """CREATE TABLE orders (
        order_id    INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL REFERENCES customers (customer_id),
        order_date  DATE NOT NULL,
        status      VARCHAR NOT NULL,
        channel     VARCHAR NOT NULL
    )""",
    """CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id      INTEGER NOT NULL REFERENCES orders (order_id),
        product_id    INTEGER NOT NULL REFERENCES products (product_id),
        quantity      INTEGER NOT NULL,
        unit_price    DECIMAL(10, 2) NOT NULL,
        discount      DECIMAL(4, 2) NOT NULL
    )""",
]

FIRST = ["Emma", "Liam", "Sophie", "Noah", "Julia", "Lucas", "Mila", "Daan", "Sara", "Finn",
         "Anna", "Sem", "Lisa", "Milan", "Eva", "Levi", "Nora", "Luuk", "Zoe", "Jesse",
         "Priya", "Arjun", "Mei", "Omar", "Elena", "Mateo", "Chloe", "Felix", "Ines", "Hugo"]
LAST = ["de Vries", "Jansen", "Bakker", "Visser", "Smit", "Meijer", "Mulder", "de Boer",
        "Schmidt", "Muller", "Martin", "Bernard", "Garcia", "Lopez", "Smith", "Brown",
        "Peeters", "Janssens", "Sharma", "Chen", "Rossi", "Novak", "Silva", "Kowalski"]
COUNTRIES = [("Netherlands", 30), ("Germany", 20), ("Belgium", 12), ("France", 12),
             ("United Kingdom", 10), ("Spain", 8), ("United States", 8)]
CATALOG = {
    "Electronics": ["Wireless Earbuds", "Smartwatch", "Bluetooth Speaker", "USB-C Hub",
                    "Mechanical Keyboard", "4K Monitor", "Webcam", "Portable SSD"],
    "Home": ["Coffee Grinder", "Desk Lamp", "Throw Blanket", "Ceramic Vase",
             "Air Purifier", "Chef's Knife", "Cast Iron Pan", "Plant Pot Set"],
    "Books": ["Data Engineering Handbook", "Designing Data Systems", "Python Cookbook",
              "SQL Puzzles", "Statistics Primer", "Cloud Patterns", "AI Engineering Notes",
              "Clean Pipelines"],
    "Sports": ["Yoga Mat", "Running Shoes", "Resistance Bands", "Water Bottle",
               "Cycling Gloves", "Foam Roller", "Jump Rope", "Gym Bag"],
    "Beauty": ["Face Serum", "Sunscreen SPF50", "Hand Cream", "Shampoo Bar",
               "Lip Balm Set", "Beard Oil", "Night Cream", "Body Lotion"],
}
PRICE_RANGE = {"Electronics": (25, 450), "Home": (12, 220), "Books": (15, 60),
               "Sports": (8, 140), "Beauty": (6, 70)}

START = date(2024, 1, 1)
END = date(2025, 12, 31)


def _weighted(rng: random.Random, pairs):
    return rng.choices([p[0] for p in pairs], weights=[p[1] for p in pairs], k=1)[0]


def generate(seed: int = 42, n_customers: int = 600, n_orders: int = 6000):
    rng = random.Random(seed)
    days = (END - START).days

    customers = []
    for cid in range(1, n_customers + 1):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        signup = START + timedelta(days=rng.randint(0, int(days * 0.55)))
        customers.append((
            cid,
            f"{first} {last}",
            f"{first.lower()}.{last.lower().replace(' ', '')}{cid}@example.com",
            _weighted(rng, COUNTRIES),
            _weighted(rng, [("consumer", 80), ("business", 20)]),
            signup.isoformat(),
        ))

    products = []
    pid = 1
    for category, names in CATALOG.items():
        lo, hi = PRICE_RANGE[category]
        for name in names:
            products.append((pid, name, category, round(rng.uniform(lo, hi), 2)))
            pid += 1

    # About 7% of customers sign up but never order.
    buyers = [c for c in customers if rng.random() > 0.07]

    orders, items = [], []
    item_id = 1
    for oid in range(1, n_orders + 1):
        cust = rng.choice(buyers)
        signup = date.fromisoformat(cust[5])
        # Later dates are more likely (business growth) and Q4 gets a bump.
        while True:
            d = signup + timedelta(days=rng.randint(0, (END - signup).days))
            growth = 0.5 + 0.5 * (d - START).days / days
            season = 1.4 if d.month in (11, 12) else 1.0
            if rng.random() < growth * season / 1.4:
                break
        status = _weighted(rng, [("completed", 85), ("cancelled", 8), ("returned", 7)])
        channel = _weighted(rng, [("web", 55), ("mobile_app", 35), ("marketplace", 10)])
        orders.append((oid, cust[0], d.isoformat(), status, channel))

        for prod in rng.sample(products, k=_weighted(rng, [(1, 55), (2, 30), (3, 15)])):
            qty = _weighted(rng, [(1, 70), (2, 20), (3, 10)])
            discount = _weighted(rng, [(0.0, 75), (0.1, 15), (0.2, 10)])
            if cust[4] == "business":
                qty *= 2
            items.append((item_id, oid, prod[0], qty, prod[3], discount))
            item_id += 1

    return {"customers": customers, "products": products, "orders": orders, "order_items": items}


def seed_database(path: str | Path, seed: int = 42, overwrite: bool = True) -> dict[str, int]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        path.unlink()
    data = generate(seed=seed)
    with Database(path, read_only=False) as db:
        db.executescript(DDL)
        for table in ("customers", "products", "orders", "order_items"):
            db.insert_many(table, data[table])
    return {table: len(rows) for table, rows in data.items()}
