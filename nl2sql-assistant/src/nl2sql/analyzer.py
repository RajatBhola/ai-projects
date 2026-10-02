"""Deterministic SQL analysis: what a query does, read straight from its syntax tree.

This is the grounding for the plain-English explanation. A parser can't be wrong
about which tables are read, how they are joined or which rows are filtered, so
the LLM is given these facts instead of having to infer them. The analyzer also
runs a few checks for classic SQL mistakes (``x = NULL``, a LEFT JOIN undone by
a WHERE filter, ``NOT IN`` with NULLs, DELETE without WHERE, ...).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

SET_OPERATIONS = (exp.Union, exp.Intersect, exp.Except)
STATEMENT_TYPES = tuple(
    t for t in (
        getattr(exp, name, None)
        for name in ("Query", "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter",
                     "Command", "TruncateTable", "Pragma", "Set", "Use", "Copy", "Attach", "Detach",
                     "Transaction", "Commit", "Rollback", "Describe", "Show")
    ) if isinstance(t, type)
)


class SQLAnalysisError(Exception):
    """The input could not be parsed as SQL."""


@dataclass
class Join:
    kind: str          # INNER, LEFT, RIGHT, FULL, CROSS
    table: str         # e.g. "customers AS c"
    condition: str     # ON/USING condition, "" if none


@dataclass
class OutputColumn:
    name: str
    expression: str


@dataclass
class Issue:
    code: str
    severity: str      # "warning" (likely a bug or risk) or "info" (worth knowing)
    message: str


@dataclass
class QueryAnalysis:
    sql: str
    dialect: str
    statement_type: str
    modifies_data: bool
    tables: list[str] = field(default_factory=list)
    target_table: str | None = None
    ctes: list[str] = field(default_factory=list)
    source: str = ""
    joins: list[Join] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    aggregations: list[str] = field(default_factory=list)
    having: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: str | None = None
    distinct: bool = False
    output_columns: list[OutputColumn] = field(default_factory=list)
    set_columns: list[str] = field(default_factory=list)   # UPDATE ... SET targets
    window_functions: list[str] = field(default_factory=list)
    subqueries: int = 0
    union_branches: int = 0
    issues: list[Issue] = field(default_factory=list)

    @property
    def has_star(self) -> bool:
        return any(c.name == "*" or c.name.endswith(".*") for c in self.output_columns)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------- dbt / Jinja
_REF = re.compile(r"\{\{\s*ref\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*['\"]([^'\"]+)['\"])?\s*\)\s*\}\}")
_SOURCE = re.compile(r"\{\{\s*source\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}")
_CONFIG = re.compile(r"\{\{\s*config\(.*?\)\s*\}\}", re.DOTALL)
_TAG = re.compile(r"\{%-?.*?-?%\}|\{#.*?#\}", re.DOTALL)
_EXPR = re.compile(r"\{\{.*?\}\}", re.DOTALL)


def strip_jinja(sql: str) -> tuple[str, bool]:
    """Make dbt-style SQL parseable: ref('x') -> x, source('s', 't') -> s.t,
    config/tags/comments removed, other {{ ... }} replaced by a placeholder.
    Returns (sql, changed)."""
    if "{{" not in sql and "{%" not in sql and "{#" not in sql:
        return sql, False
    out = _REF.sub(lambda m: m.group(2) or m.group(1), sql)
    out = _SOURCE.sub(lambda m: f"{m.group(1)}.{m.group(2)}", out)
    out = _CONFIG.sub("", out)
    out = _TAG.sub("", out)
    out = _EXPR.sub("jinja_value", out)
    return out.strip(), True


# ----------------------------------------------------------------------------- helpers
def _sql(node: exp.Expression | None, dialect: str) -> str:
    return node.sql(dialect=dialect) if node is not None else ""


def _arg(node: exp.Expression, *names: str):
    for name in names:
        value = node.args.get(name)
        if value is not None:
            return value
    return None


def _conditions(clause: exp.Expression | None) -> list[exp.Expression]:
    """Split a WHERE/HAVING condition on top-level ANDs."""
    if clause is None:
        return []
    cond = clause.this if isinstance(clause, (exp.Where, exp.Having)) else clause
    while isinstance(cond, exp.Paren) and isinstance(cond.this, exp.And):
        cond = cond.this
    if isinstance(cond, exp.And):
        return list(cond.flatten())
    return [cond]


def _table_label(node: exp.Expression, dialect: str) -> str:
    if isinstance(node, exp.Table):
        name = ".".join(p for p in (node.catalog, node.db, node.name) if p)
        return f"{name} AS {node.alias}" if node.alias else name
    if isinstance(node, exp.Subquery):
        return f"(subquery) AS {node.alias}" if node.alias else "(subquery)"
    return _sql(node, dialect)


def _alias(node: exp.Expression) -> str:
    return (node.alias or (node.name if isinstance(node, exp.Table) else "")).lower()


def _join_kind(join: exp.Join) -> str:
    side, kind = (join.side or "").upper(), (join.kind or "").upper()
    if kind == "CROSS":
        return "CROSS"
    return side if side in ("LEFT", "RIGHT", "FULL") else "INNER"


def _statement_type(tree: exp.Expression) -> str:
    if isinstance(tree, exp.Union) and not isinstance(tree, (exp.Intersect, exp.Except)):
        return "UNION" if tree.args.get("distinct") else "UNION ALL"
    if isinstance(tree, (exp.Create, exp.Drop)):
        kind = tree.args.get("kind")
        return f"{tree.key.upper()} {kind}" if kind else tree.key.upper()
    return {"select": "SELECT", "truncatetable": "TRUNCATE"}.get(tree.key, tree.key.upper())


def _first_select(tree: exp.Expression) -> exp.Select | None:
    node = tree
    while isinstance(node, SET_OPERATIONS):
        node = node.this
    if isinstance(node, exp.Subquery):
        node = node.this
    return node if isinstance(node, exp.Select) else None


# ----------------------------------------------------------------------------- analysis
def parse(sql: str, dialect: str = "duckdb") -> list[exp.Expression]:
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except ParseError as e:
        detail = e.errors[0] if e.errors else {}
        where = f" (line {detail.get('line')}, column {detail.get('col')})" if detail.get("line") else ""
        raise SQLAnalysisError(f"Could not parse the SQL{where}: {detail.get('description', e)}") from e
    if not statements:
        raise SQLAnalysisError("No SQL statement found.")
    for s in statements:
        if not isinstance(s, STATEMENT_TYPES):
            raise SQLAnalysisError(f"This doesn't look like a SQL statement: {s.sql()[:80]!r}")
    return statements


def analyze(sql: str, dialect: str = "duckdb") -> list[QueryAnalysis]:
    """Analyze every statement in ``sql``."""
    cleaned, had_jinja = strip_jinja(sql)
    results = [analyze_statement(tree, dialect) for tree in parse(cleaned, dialect)]
    if had_jinja:
        for a in results:
            a.issues.insert(0, Issue("jinja", "info",
                                     "dbt/Jinja templating was simplified before analysis "
                                     "(ref() and source() became table names)."))
    return results


def analyze_statement(tree: exp.Expression, dialect: str = "duckdb") -> QueryAnalysis:
    a = QueryAnalysis(
        sql=tree.sql(dialect=dialect, pretty=True),
        dialect=dialect,
        statement_type=_statement_type(tree),
        modifies_data=not isinstance(tree, exp.Query),
    )
    cte_names = [cte.alias_or_name for cte in tree.find_all(exp.CTE)]
    a.ctes = cte_names
    seen = []
    for t in tree.find_all(exp.Table):
        name = ".".join(p for p in (t.catalog, t.db, t.name) if p)
        if name and name.lower() not in {c.lower() for c in cte_names} and name not in seen:
            seen.append(name)
    a.tables = seen

    if isinstance(tree, (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create)):
        target = tree.find(exp.Table)
        a.target_table = target.name if target is not None else None

    main: exp.Expression | None
    if isinstance(tree, exp.Query):
        main = _first_select(tree)
        if isinstance(tree, SET_OPERATIONS):
            a.union_branches = sum(1 for s in tree.find_all(exp.Select)
                                   if isinstance(s.parent, SET_OPERATIONS))
    elif isinstance(tree, exp.Insert):
        main = _first_select(tree.expression) if tree.expression is not None else None
    else:
        main = None

    if isinstance(tree, (exp.Update, exp.Delete)):
        a.filters = [_sql(c, dialect) for c in _conditions(tree.args.get("where"))]
    if isinstance(tree, exp.Update):
        a.set_columns = [_sql(e.this, dialect) for e in tree.expressions if isinstance(e, exp.EQ)]

    if main is not None:
        _describe_select(main, a, dialect)

    a.subqueries = sum(
        1 for s in tree.find_all(exp.Select)
        if s is not main and s is not tree
        and not isinstance(s.parent, (exp.CTE, exp.Create, *SET_OPERATIONS))
        and not isinstance(s.parent, exp.Insert)
    )
    a.issues = _check(tree, main, a, dialect)
    return a


def _describe_select(select: exp.Select, a: QueryAnalysis, dialect: str) -> None:
    source = _arg(select, "from", "from_")
    if source is not None:
        a.source = _table_label(source.this, dialect)
    for join in select.args.get("joins") or []:
        on = join.args.get("on")
        using = join.args.get("using")
        if on is not None:
            condition = _sql(on, dialect)
        elif using:
            condition = "USING (" + ", ".join(_sql(u, dialect) for u in using) + ")"
        else:
            condition = ""
        a.joins.append(Join(kind=_join_kind(join), table=_table_label(join.this, dialect), condition=condition))

    a.filters = [_sql(c, dialect) for c in _conditions(select.args.get("where"))]
    a.having = [_sql(c, dialect) for c in _conditions(select.args.get("having"))]
    group = select.args.get("group")
    a.group_by = [_sql(g, dialect) for g in group.expressions] if group is not None else []
    order = select.args.get("order")
    a.order_by = [_sql(o, dialect) for o in order.expressions] if order is not None else []
    limit = select.args.get("limit")
    if limit is not None:
        a.limit = _sql(limit.expression, dialect) if limit.expression is not None else _sql(limit, dialect)
    a.distinct = select.args.get("distinct") is not None

    aggs, windows = [], []
    for e in select.expressions + ([select.args["having"]] if select.args.get("having") else []):
        for w in e.find_all(exp.Window):
            text = _sql(w, dialect)
            if text not in windows:
                windows.append(text)
        for agg in e.find_all(exp.AggFunc):
            if agg.find_ancestor(exp.Window, exp.Subquery) is None:
                text = _sql(agg, dialect)
                if text not in aggs:
                    aggs.append(text)
    a.aggregations, a.window_functions = aggs, windows

    for e in select.expressions:
        if isinstance(e, exp.Star):
            a.output_columns.append(OutputColumn("*", "*"))
        elif isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
            a.output_columns.append(OutputColumn(f"{e.table}.*", _sql(e, dialect)))
        else:
            expression = _sql(e.unalias(), dialect)
            a.output_columns.append(OutputColumn(e.output_name or expression, expression))

    # GROUP BY 1 / ORDER BY 2 refer to output columns by position: show the name.
    def resolve(item: str) -> str:
        head, _, rest = item.partition(" ")
        if head.isdigit() and 1 <= int(head) <= len(a.output_columns):
            return " ".join(p for p in (a.output_columns[int(head) - 1].name, rest) if p)
        return item

    a.group_by = [resolve(g) for g in a.group_by]
    a.order_by = [resolve(o) for o in a.order_by]


# ----------------------------------------------------------------------------- checks
def _check(tree: exp.Expression, main: exp.Select | None, a: QueryAnalysis, dialect: str) -> list[Issue]:
    issues: list[Issue] = []

    if a.modifies_data:
        issues.append(Issue("modifies_data", "warning",
                            f"This is a {a.statement_type} statement: it changes the database "
                            "rather than just reading it."))
    if isinstance(tree, (exp.Update, exp.Delete)) and tree.args.get("where") is None:
        issues.append(Issue("no_where", "warning",
                            f"There is no WHERE clause, so this {a.statement_type} affects every row "
                            f"in {a.target_table}."))

    for cmp in tree.find_all(exp.EQ, exp.NEQ):
        if isinstance(cmp.this, exp.Null) or isinstance(cmp.expression, exp.Null):
            issues.append(Issue("null_comparison", "warning",
                                f"`{_sql(cmp, dialect)}` is never true, because nothing equals NULL. "
                                "Use IS NULL or IS NOT NULL instead."))

    for neg in tree.find_all(exp.Not):
        if isinstance(neg.this, exp.In) and neg.this.args.get("query") is not None:
            issues.append(Issue("not_in_subquery", "warning",
                                "NOT IN with a subquery returns no rows at all if the subquery returns "
                                "any NULL. NOT EXISTS is the safer pattern."))

    if isinstance(tree, exp.Union) and not isinstance(tree, (exp.Intersect, exp.Except)) \
            and tree.args.get("distinct"):
        issues.append(Issue("union_distinct", "info",
                            "UNION removes duplicate rows, which costs extra work. If duplicates are "
                            "impossible or wanted, UNION ALL is faster."))

    if main is not None:
        issues.extend(_check_select(main, a, dialect))
    return issues


def _check_select(select: exp.Select, a: QueryAnalysis, dialect: str) -> list[Issue]:
    issues: list[Issue] = []
    where = select.args.get("where")
    conditions = _conditions(where)

    if a.has_star:
        issues.append(Issue("select_star", "info",
                            "SELECT * returns every column, including any added to the table later. "
                            "Listing the columns you need is safer and often faster."))

    if not a.modifies_data and not a.limit and not a.group_by and not a.aggregations \
            and not a.distinct and a.source:
        issues.append(Issue("no_limit", "info",
                            "There is no LIMIT, so this returns every matching row, which could be a lot."))

    for join in select.args.get("joins") or []:
        alias = _alias(join.this)
        kind = _join_kind(join)
        has_condition = join.args.get("on") is not None or bool(join.args.get("using"))
        if not has_condition and kind != "CROSS" and not _linked_in_where(alias, conditions):
            issues.append(Issue("missing_join_condition", "warning",
                                f"{_table_label(join.this, dialect)} is joined without a condition, so "
                                "every row is paired with every other row (a cartesian product)."))
        if kind == "LEFT" and alias:
            for cond in conditions:
                if cond.find(exp.Is) is not None:
                    continue  # IS NULL / IS NOT NULL checks are deliberate
                cols = [c for c in cond.find_all(exp.Column) if (c.table or "").lower() == alias]
                if cols:
                    issues.append(Issue("left_join_filtered", "warning",
                                        f"The filter `{_sql(cond, dialect)}` is on the LEFT JOINed table "
                                        f"{alias}, which drops the unmatched rows the LEFT JOIN was meant "
                                        "to keep, so it behaves like an INNER JOIN. Move it into the ON "
                                        "clause if that isn't intended."))
                    break
    return issues


def _linked_in_where(alias: str, conditions: list[exp.Expression]) -> bool:
    """Is the comma-joined table tied to another table by an equality in WHERE?"""
    for cond in conditions:
        for eq in cond.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                sides = {(left.table or "").lower(), (right.table or "").lower()}
                if alias in sides and len(sides) == 2:
                    return True
    return False


# ----------------------------------------------------------------------------- offline description
def _order_text(item: str) -> str:
    if item.upper().endswith(" DESC"):
        return f"{item[:-5]} (highest first)"
    if item.upper().endswith(" ASC"):
        return f"{item[:-4]} (lowest first)"
    return f"{item} (lowest first)"


JOIN_TEXT = {
    "INNER": "keeping only rows that match on both sides",
    "LEFT": "keeping every row from the left side, even without a match",
    "RIGHT": "keeping every row from the right side, even without a match",
    "FULL": "keeping all rows from both sides, matched where possible",
    "CROSS": "pairing every row with every row",
}


def describe(a: QueryAnalysis) -> list[str]:
    """A template-based, plain-English walk through the query (no LLM needed)."""
    steps: list[str] = []
    if a.statement_type == "DELETE":
        where = f" where {' and '.join(a.filters)}" if a.filters else " (every row)"
        return [f"Deletes rows from {a.target_table}{where}."]
    if a.statement_type == "UPDATE":
        where = f" where {' and '.join(a.filters)}" if a.filters else " in every row"
        return [f"Changes {', '.join(a.set_columns)} in {a.target_table}{where}."]
    if a.modifies_data and a.statement_type != "INSERT":
        return [f"Runs a {a.statement_type} statement on {a.target_table or ', '.join(a.tables)}."]
    if a.statement_type == "INSERT":
        steps.append(f"Adds rows to {a.target_table}, taken from the following query:")

    if a.ctes:
        steps.append(f"First builds {len(a.ctes)} temporary result(s): {', '.join(a.ctes)}.")
    if a.source:
        steps.append(f"Reads from {a.source}.")
    for j in a.joins:
        if j.condition:
            steps.append(f"Combines it with {j.table} on {j.condition}, {JOIN_TEXT[j.kind]}.")
        elif j.kind == "CROSS":
            steps.append(f"Combines it with {j.table}, {JOIN_TEXT['CROSS']}.")
        else:
            steps.append(f"Combines it with {j.table} (no join condition here; any link is in the filters below).")
    if a.filters:
        steps.append("Keeps only rows where " + "; and ".join(a.filters) + ".")
    if a.group_by:
        calc = f" and for each group calculates {', '.join(a.aggregations)}" if a.aggregations else ""
        steps.append(f"Groups the rows by {', '.join(a.group_by)}{calc}.")
    elif a.aggregations:
        steps.append(f"Calculates {', '.join(a.aggregations)} over all those rows.")
    if a.having:
        steps.append("Keeps only groups where " + "; and ".join(a.having) + ".")
    if a.window_functions:
        steps.append(f"Adds running/ranked values: {', '.join(a.window_functions)}.")
    if a.distinct:
        steps.append("Removes duplicate rows.")
    if a.union_branches:
        steps.append(f"Stacks the results of {a.union_branches} queries together ({a.statement_type}).")
    if a.order_by:
        steps.append("Sorts by " + ", then ".join(_order_text(o) for o in a.order_by) + ".")
    if a.limit:
        steps.append(f"Returns at most {a.limit} rows.")
    return steps
