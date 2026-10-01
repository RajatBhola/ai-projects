"""Command-line interface.

    nl2sql seed                      # build the sample warehouse
    nl2sql ask "revenue by country"  # ask a question
    nl2sql context "revenue by country"   # show what the model would see (no API call)
    nl2sql eval                      # run the evaluation set
"""

from __future__ import annotations

import argparse
import sys
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
    p = sub.add_parser("context", help="print the schema context for a question (no API call)")
    p.add_argument("question")
    p = sub.add_parser("eval", help="run the evaluation set")
    p.add_argument("--questions", default=str(PROJECT_ROOT / "evals" / "questions.yaml"))
    p.add_argument("--only", help="comma-separated case ids")
    p.add_argument("--out", help="where to save the JSON report")

    args = parser.parse_args(argv)
    handler = {"seed": cmd_seed, "ask": cmd_ask, "context": cmd_context, "eval": cmd_eval}[args.command]
    return handler(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
