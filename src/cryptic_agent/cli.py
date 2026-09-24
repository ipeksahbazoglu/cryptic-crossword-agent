"""Command-line entry point: `cryptic-agent <command> [options]`.

Each pipeline stage is a subcommand. Handlers take the parsed arguments and
return a process exit code; stage logic lives in its own module, not here.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from cryptic_agent import config

Handler = Callable[[argparse.Namespace], int]


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"'{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cryptic-agent",
        description="Scrape, extract, solve and evaluate UK cryptic crossword clues.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    scrape = subparsers.add_parser("scrape", help="Fetch Fifteensquared blog posts.")
    scrape.add_argument("--category", required=True, help="Category slug, e.g. 'guardian'.")
    scrape.add_argument("--pages", type=int, default=5, help="Number of pages to fetch.")
    scrape.add_argument("--per-page", type=int, default=20, help="Posts per page.")
    scrape.set_defaults(handler=_not_implemented)

    extract = subparsers.add_parser("extract", help="Turn raw posts into structured clues.")
    extract.add_argument("--input", type=Path, default=None, help="Raw posts directory.")
    extract.add_argument("--output", type=Path, default=None, help="Output .jsonl path.")
    extract.add_argument("--limit", type=int, default=None, help="Max posts to process.")
    extract.set_defaults(handler=_not_implemented)

    solve = subparsers.add_parser("solve", help="Solve a single clue.")
    solve.add_argument("--clue", required=True, help="Clue text without the enumeration.")
    solve.add_argument("--enumeration", required=True, help="Answer lengths, e.g. '7' or '3,4'.")
    solve.add_argument("--examples", type=Path, default=None, help="clues.jsonl for few-shot.")
    solve.add_argument("--n-examples", type=int, default=5, help="Few-shot examples to include.")
    solve.set_defaults(handler=_not_implemented)

    evaluate = subparsers.add_parser("eval", help="Score the solver on held-out clues.")
    evaluate.add_argument("--dataset", type=Path, default=None, help="clues.jsonl to test on.")
    evaluate.add_argument("--n", type=int, default=50, help="Number of test clues.")
    evaluate.add_argument("--n-examples", type=int, default=5, help="Few-shot examples per clue.")
    evaluate.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    evaluate.set_defaults(handler=_not_implemented)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. `argv` defaults to sys.argv[1:]; tests pass it explicitly."""
    load_dotenv(find_dotenv(usecwd=True))  # .env from where you run the command
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    try:
        return handler(args)
    except config.MissingAPIKeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
