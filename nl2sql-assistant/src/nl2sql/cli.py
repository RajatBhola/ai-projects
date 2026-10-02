"""Command-line interface.

    nl2sql seed                      # build the sample warehouse
    nl2sql ask "revenue by country"  # ask a question
    nl2sql explain "SELECT ..."      # explain a SQL query in plain English
    nl2sql context "revenue by country"   # show what the model would see (no API call)
    nl2sql eval                      # run the evaluation set
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT, Settings
from .database import Database
from .semantic import SemanticLayer


def format_table(columns: list[str], rows: list[tuple], max_rows: int = 25) -> str:
    if not columns:
        return "(no columns)"
    shown = [[_fmt(v) for v in r] for r in rows[:max_rows]]
    widths = [max(len(c), *(len(r[i]) for r in shown)) if shown else len(c) for i, c in enumerate(columns)]
    line = "  ".join(c.ljust(w) for c, w in zip(columns, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(v.ljust(w) for v, w in zip(r, widths)) for r in shown]
    if len(rows) > max_rows:
        out.append(f"... {len(rows) - max_rows} more rows")
    return "\n".join(out)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}"
    return "NULL" if v is None else str(v)


def _pipeline(settings: Settings, db_path: Path):
    from .llm import OpenAIClient
    from .pipeline import NL2SQL

    db = Database(db_path)
    semantic = SemanticLayer.load(settings.semantic_layer_path)
    llm = OpenAIClient(model=settings.openai_model)
    return NL2SQL(db, llm, semantic, max_rows=settings.max_rows, max_attempts=settings.max_attempts)


def cmd_seed(args, settings: Settings) -> int:
    from .seed import seed_database

    counts = seed_database(args.db)
    print(f"Created {args.db}")
    for table, n in counts.items():
        print(f"  {table:<12} {n:>6} rows")
    return 0


def cmd_context(args, settings: Settings) -> int:
    from .schema_context import build_context

    with Database(args.db) as db:
        ctx = build_context(db, SemanticLayer.load(settings.semantic_layer_path), args.question, max_tables=6)
    print(ctx.text)
    return 0


def cmd_ask(args, settings: Settings) -> int:
    nl = _pipeline(settings, args.db)
    answer = nl.ask(args.question)
    for i, att in enumerate(answer.attempts, 1):
        if att.error:
            print(f"Attempt {i} rejected ({att.stage}): {att.error}", file=sys.stderr)
    if not answer.ok:
        print(f"Could not answer: {answer.error}")
        return 1
    print(f"\n{answer.explanation}")
    for a in answer.assumptions:
        print(f"  assumption: {a}")
    print(f"\n{answer.sql}\n")
    print(format_table(answer.columns, answer.rows))
    print(f"\n{len(answer.rows)} rows in {answer.seconds}s, {len(answer.attempts)} attempt(s), "
          f"{answer.input_tokens + answer.output_tokens} tokens")
    return 0


def _read_sql(args) -> str:
    if args.file:
        return args.file.read_text()
    if args.sql and args.sql != "-":
        return args.sql
    if args.sql == "-" or not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def _wrap(text: str, indent: str = "  ", first: str | None = None) -> str:
    return textwrap.fill(text, width=88, initial_indent=first if first is not None else indent,
                         subsequent_indent=indent)


def render_explanation(ex, index: int | None = None) -> str:
    a = ex.analysis
    out = []
    if index is not None:
        out.append(f"=== Statement {index} ===")
    out.append(_wrap(ex.summary, indent="", first=""))
    if ex.steps:
        out.append("\nStep by step")
        out.extend(_wrap(s, indent="     ", first=f"  {i}. ") for i, s in enumerate(ex.steps, 1))
    if ex.output_columns:
        out.append("\nColumns returned")
        width = min(max(len(c.name) for c in ex.output_columns), 28)
        for c in ex.output_columns:
            name = c.name if len(c.name) <= width else c.name[: width - 1] + "…"
            out.append(_wrap(c.meaning or "", indent=" " * (width + 6), first=f"  {name.ljust(width)}    ")
                       if c.meaning else f"  {name}")
    warnings = [i for i in a.issues if i.severity == "warning"]
    notes = [i for i in a.issues if i.severity == "info"]
    if warnings or ex.caveats:
        out.append("\nWatch out")
        out.extend(_wrap(i.message, indent="    ", first="  ! ") for i in warnings)
        out.extend(_wrap(c, indent="    ", first="  ! ") for c in ex.caveats)
    if notes:
        out.append("\nNotes")
        out.extend(_wrap(i.message, indent="    ", first="  - ") for i in notes)
    footer = f"\n{a.statement_type} · {'changes data' if a.modifies_data else 'read-only'}"
    if a.tables:
        footer += f" · tables: {', '.join(a.tables)}"
    if ex.source == "llm":
        footer += f" · {ex.input_tokens + ex.output_tokens} tokens"
    else:
        footer += " · offline (no LLM)"
    out.append(footer)
    return "\n".join(out)


def cmd_explain(args, settings: Settings) -> int:
    import sqlglot

    from .analyzer import SQLAnalysisError
    from .explainer import explain

    sql = _read_sql(args)
    if not sql.strip():
        print("No SQL given. Pass it as an argument, with --file, or on stdin.", file=sys.stderr)
        return 2
    dialect = args.dialect or ("duckdb" if settings.db_path.suffix == ".duckdb" else "sqlite")
    try:
        sqlglot.Dialect.get_or_raise(dialect)
    except ValueError:
        print(f"Unknown dialect '{dialect}'.", file=sys.stderr)
        return 2

    semantic = SemanticLayer.empty()
    if not args.no_semantic_layer and settings.semantic_layer_path.is_file():
        semantic = SemanticLayer.load(settings.semantic_layer_path)

    llm = None
    if not args.offline:
        try:
            from .llm import OpenAIClient

            llm = OpenAIClient(model=settings.openai_model)
        except Exception as e:  # noqa: BLE001 - missing package or API key
            print(f"Can't use the LLM ({e}).\nSet OPENAI_API_KEY in .env, or run with --offline "
                  "for a structural explanation.", file=sys.stderr)
            return 1

    try:
        explanations = explain(sql, llm=llm, semantic=semantic, dialect=dialect, audience=args.audience)
    except SQLAnalysisError as e:
        print(str(e), file=sys.stderr)
        return 2

    if args.json:
        data = [ex.to_dict() for ex in explanations]
        print(json.dumps(data[0] if len(data) == 1 else data, indent=2, default=str))
        return 0
    multiple = len(explanations) > 1
    print("\n\n".join(render_explanation(ex, i if multiple else None)
                      for i, ex in enumerate(explanations, 1)))
    return 0


def cmd_eval(args, settings: Settings) -> int:
    from .evaluation import load_cases, run_eval

    nl = _pipeline(settings, args.db)
    cases = load_cases(args.questions)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only.split(","))]
    report = run_eval(nl, cases)
    for r in report.results:
        mark = "PASS" if r.correct else "FAIL"
        print(f"{mark}  {r.id:<4} {r.question[:60]:<60}  {'' if r.correct else r.reason}")
    print(f"\nAccuracy: {report.correct}/{report.total} = {report.accuracy:.1%}")
    for level, s in sorted(report.by_difficulty.items()):
        print(f"  {level:<7} {s['correct']}/{s['total']}")
    print(f"Valid on first try: {report.first_try_valid_rate:.1%}   avg attempts: {report.avg_attempts}   "
          f"avg latency: {report.avg_seconds}s   tokens: {report.total_input_tokens + report.total_output_tokens}")
    out = Path(args.out) if args.out else PROJECT_ROOT / "evals" / "results" / (
        f"{datetime.now():%Y%m%d-%H%M%S}-{settings.openai_model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.to_json())
    print(f"Saved {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(prog="nl2sql", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, default=settings.db_path, help="database file (.duckdb or .sqlite)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="create the sample warehouse")
    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p = sub.add_parser("explain", help="explain a SQL query in plain English",
                       description="Explain what a SQL query does, in plain English. Nothing is executed.")
    p.add_argument("sql", nargs="?", help="the SQL (or '-' to read stdin); omit when using --file")
    p.add_argument("-f", "--file", type=Path, help="read the SQL from a file")
    p.add_argument("--dialect", default=None,
                   help="SQL dialect to parse: duckdb, postgres, snowflake, bigquery, tsql, ... "
                        "(default: the configured database's dialect)")
    p.add_argument("--audience", choices=["business", "technical"], default="business",
                   help="who the explanation is for (default: business)")
    p.add_argument("--offline", action="store_true",
                   help="no LLM call: structural explanation and checks only (no API key needed)")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--no-semantic-layer", action="store_true",
                   help="don't use semantic_layer.yaml for table and column meanings")
    p = sub.add_parser("context", help="print the schema context for a question (no API call)")
    p.add_argument("question")
    p = sub.add_parser("eval", help="run the evaluation set")
    p.add_argument("--questions", default=str(PROJECT_ROOT / "evals" / "questions.yaml"))
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--out", help="where to save the JSON report")

    args = parser.parse_args(argv)
    handler = {"seed": cmd_seed, "ask": cmd_ask, "explain": cmd_explain,
               "context": cmd_context, "eval": cmd_eval}[args.command]
    return handler(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
