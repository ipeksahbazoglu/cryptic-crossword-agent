"""Command-line entry point: `cryptic-agent <command> [options]`.

Each pipeline stage is a subcommand. Handlers take the parsed arguments and
return a process exit code; stage logic lives in its own module, not here.
"""

import argparse
import logging
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from cryptic_agent import config
from cryptic_agent.agent.solver import Solver, Step
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.evaluation.run import (
    ResultRow,
    clue_key,
    format_report,
    load_clues,
    load_posts,
    run_evaluation,
    sample_clues,
    summarise,
)
from cryptic_agent.jsonl import read_jsonl
from cryptic_agent.lexicon.build import build_lexicon
from cryptic_agent.lexicon.sources import CRYPTICS, MOBY, SOURCES, ChecksumMismatchError, fetch
from cryptic_agent.lexicon.store import Lexicon, LexiconNotFoundError
from cryptic_agent.lexicon.store import default_path as lexicon_path
from cryptic_agent.llm.client import GroqClient, RateLimitedError
from cryptic_agent.scraper.client import MAX_PER_PAGE, CategoryNotFoundError, WordPressClient
from cryptic_agent.scraper.scrape import output_path, scrape_category
from cryptic_agent.tools.dictionary import Dictionary, DictionaryNotFoundError

Handler = Callable[[argparse.Namespace], int]


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


def _scrape(args: argparse.Namespace) -> int:
    out_path = args.output or output_path(config.raw_dir(), args.category)
    try:
        summary = scrape_category(
            WordPressClient(), args.category, max_pages=args.pages, out_path=out_path
        )
    except CategoryNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Fetched {summary.fetched} posts ({summary.new} new). "
        f"{summary.total} posts in {summary.path}"
    )
    return 0


def _ingest(args: argparse.Namespace) -> int:
    sources_dir = config.lexicon_dir() / "sources"
    for name in args.source or list(SOURCES):
        source = SOURCES[name]
        try:
            path = fetch(source, sources_dir)
        except ChecksumMismatchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"{source.name}: {path}\n  licence: {source.licence}")

    if all((sources_dir / s.filename).exists() for s in (MOBY, CRYPTICS)):
        print("building the lexicon (about a minute)...")
        summary = build_lexicon(sources_dir, lexicon_path())
        print(f"lexicon: {summary.path}")
        for table, count in summary.counts.items():
            print(f"  {table:16s} {count:>10,}")
    else:
        print(
            "lexicon not built: needs the moby and cryptics sources (run ingest without --source)"
        )
    return 0


def _print_step(step: Step) -> None:
    if step.kind == "fastpass":
        print(textwrap.indent(step.text, "  "))
    elif step.kind == "tier":
        print(f"-- {step.text}")
    elif step.kind == "thought":
        print(f"  thinking: {textwrap.shorten(step.text, 300)}")
    elif step.kind == "tool":
        args = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
        print(f"  tool: {step.tool}({args})\n        -> {textwrap.shorten(step.result, 200)}")


def _solve(args: argparse.Namespace) -> int:
    try:
        toolbox = Toolbox(Dictionary.load(), Lexicon())
    except (LexiconNotFoundError, DictionaryNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    llm = GroqClient()
    solver = Solver(
        llm,
        toolbox,
        on_step=None if args.quiet else _print_step,
        fast_pass=not args.no_fast_pass,
        strategy=args.strategy,
    )
    print(f"Clue: {args.clue} ({args.enumeration})")
    result = solver.solve(args.clue, args.enumeration, pattern=args.pattern)

    if result.verdict is None or result.worksheet is None:
        print(f"\nNo valid worksheet: {result.error}", file=sys.stderr)
        return 1
    verdict, worksheet = result.verdict, result.worksheet
    tier = {0: "code only", 1: "one model call", 2: "the agent"}.get(result.tier or 0, "")
    print(f"\nANSWER: {verdict.answer}  [{verdict.status.upper()}]  (tier {result.tier}: {tier})")
    for d in worksheet.definitions:
        print(f"  definition: {d.text!r} ({d.position})")
    for w in worksheet.wordplay:
        indicator = f", indicator {w.indicator!r}" if w.indicator else ""
        print(f"  wordplay:   {w.mechanism} of {w.fodder!r}{indicator} -> {w.produces}")
        print(f"              {w.explanation}")
    if worksheet.link_words:
        print(f"  link words: {', '.join(repr(w) for w in worksheet.link_words)}")
    if worksheet.alternatives:
        print(f"  alternatives: {', '.join(worksheet.alternatives)}")
    print("checks:")
    for c in verdict.checks:
        mark = {True: "ok", False: "FAIL", None: "?"}[c.passed]
        print(f"  [{mark:>4}] {c.name}: {c.detail}")
    print(
        f"cost: {llm.totals.requests} requests, {result.usage.total_tokens:,} tokens, "
        f"waited {llm.totals.waited_seconds:.0f}s for rate limits"
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    name = args.name or f"n{args.n}-seed{args.seed}-{args.strategy}" + (
        f"-reveal{args.reveal_every}" if args.reveal_every else ""
    )
    out_path = config.data_dir() / "runs" / f"{name}.jsonl"
    posts_path = output_path(config.raw_dir(), "guardian/quick-cryptic")
    if not posts_path.exists():
        print(f"error: no scraped clues at {posts_path}; run cryptic-agent scrape", file=sys.stderr)
        return 2
    clues = sample_clues(load_clues(load_posts(posts_path)), args.n, args.seed)

    if not args.report_only:
        try:
            toolbox = Toolbox(Dictionary.load(), Lexicon())
        except (LexiconNotFoundError, DictionaryNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        llm = GroqClient()

        def progress(row: ResultRow, i: int, total: int) -> None:
            mark = "right" if row.right else "WRONG"
            print(
                f"[{i}/{total}] {row.status:<9} {mark}  {row.answer or '-':<12} "
                f"({row.expected})  {row.clue} ({row.enumeration})",
                flush=True,
            )

        try:
            run_evaluation(
                clues,
                lambda tb: Solver(llm, tb, fast_pass=not args.no_fast_pass, strategy=args.strategy),
                toolbox,
                out_path,
                reveal_every=args.reveal_every,
                on_row=progress,
            )
        except RateLimitedError as exc:
            print(
                f"\nStopped: out of Groq quota ({str(exc)[:160]}...).\n"
                "Progress is saved; run the same command again later to resume.",
                file=sys.stderr,
            )
    if not out_path.exists():
        print(f"error: no results at {out_path}", file=sys.stderr)
        return 2
    keys = {clue_key(c) for c in clues}
    rows = [r for r in read_jsonl(out_path, ResultRow) if r.key in keys]
    print(f"\nRun: {out_path} ({len(rows)}/{len(clues)} clues)\n")
    print(format_report(summarise(rows)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptic-agent",
        description="Scrape, extract, solve and evaluate UK cryptic crossword clues.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    scrape = subparsers.add_parser("scrape", help="Fetch Fifteensquared blog posts.")
    scrape.add_argument(
        "--category", required=True, help="Category slug, e.g. 'guardian/quick-cryptic'."
    )
    scrape.add_argument(
        "--pages", type=int, default=1, help=f"Pages to fetch, newest first ({MAX_PER_PAGE} each)."
    )
    scrape.add_argument("--output", type=Path, default=None, help="Output .jsonl path.")
    scrape.set_defaults(handler=_scrape)

    ingest = subparsers.add_parser("ingest", help="Download reference data (word lists etc.).")
    ingest.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        help="Only this source (repeatable). Default: all.",
    )
    ingest.set_defaults(handler=_ingest)

    extract = subparsers.add_parser("extract", help="Turn raw posts into structured clues.")
    extract.add_argument("--input", type=Path, default=None, help="Raw posts directory.")
    extract.add_argument("--output", type=Path, default=None, help="Output .jsonl path.")
    extract.add_argument("--limit", type=int, default=None, help="Max posts to process.")
    extract.set_defaults(handler=_not_implemented)

    solve = subparsers.add_parser("solve", help="Solve a single clue.")
    solve.add_argument("--clue", required=True, help="Clue text without the enumeration.")
    solve.add_argument("--enumeration", required=True, help="Answer lengths, e.g. '7' or '3,4'.")
    solve.add_argument(
        "--pattern", default=None, help="Known letters from crossing answers, e.g. 'T?E?S?N'."
    )
    solve.add_argument("--quiet", action="store_true", help="Only print the verdict.")
    solve.add_argument(
        "--no-fast-pass", action="store_true", help="Skip the mechanical candidate search."
    )
    solve.add_argument(
        "--strategy",
        choices=["tiered", "agent"],
        default="tiered",
        help="tiered: code, then one call, then the agent (default). agent: always the agent.",
    )
    solve.set_defaults(handler=_solve)

    evaluate = subparsers.add_parser("eval", help="Score the solver on real scraped clues.")
    evaluate.add_argument("--n", type=int, default=50, help="Number of clues to solve.")
    evaluate.add_argument("--seed", type=int, default=42, help="Seed for the sample.")
    evaluate.add_argument(
        "--name", default=None, help="Run name; results go to data/runs/<name>.jsonl."
    )
    evaluate.add_argument(
        "--reveal-every",
        type=int,
        default=None,
        help="Simulate crossing letters: reveal every Nth letter (2 = M?R?N?U?S).",
    )
    evaluate.add_argument(
        "--report-only", action="store_true", help="Summarise an existing run, solve nothing."
    )
    evaluate.add_argument(
        "--no-fast-pass", action="store_true", help="Skip the mechanical candidate search."
    )
    evaluate.add_argument(
        "--strategy",
        choices=["tiered", "agent"],
        default="tiered",
        help="tiered: code, then one call, then the agent (default). agent: always the agent.",
    )
    evaluate.set_defaults(handler=_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. `argv` defaults to sys.argv[1:]; tests pass it explicitly."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(find_dotenv(usecwd=True))  # .env from where you run the command
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    try:
        return handler(args)
    except config.MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
