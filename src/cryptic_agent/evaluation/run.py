"""Score the solver on real clues, reproducibly and fairly.

    sample  a seeded, stratified sample of scraped clues (blogger's answer known)
    run     solve each one, appending a ResultRow to a JSONL file as it goes, so an
            interrupted run resumes where it stopped (the free tier makes a
            100-clue run take about an hour)
    report  accuracy overall and by clue type, and the number that matters most:
            how many CONFIRMED answers were wrong

Fairness: each clue's own puzzle is hidden from every lexicon lookup
(`exclude_urls`), so the solver can never look up the clue it is being tested on.
"""

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from cryptic_agent.agent.solver import Solver, SolveResult
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.extraction.table_parser import ParsedClue, parse_clue_table
from cryptic_agent.jsonl import append_jsonl, read_jsonl
from cryptic_agent.models import RawPost

logger = logging.getLogger(__name__)


class ResultRow(BaseModel):
    """One evaluated clue."""

    model_config = ConfigDict(frozen=True)

    key: str  # source_url#number-direction: identifies the clue across runs
    clue: str
    enumeration: str
    kind: str  # the blogger's wordplay type hint, or "unmarked"
    expected: str
    answer: str | None
    status: str  # confirmed | pencilled | unsure | error
    right: bool
    tokens: int
    tool_calls: int
    seconds: float
    pattern: str | None = None
    error: str = ""


def clue_key(clue: ParsedClue) -> str:
    return f"{clue.source_url}#{clue.number}-{clue.direction or '?'}"


def kind_of(clue: ParsedClue) -> str:
    return clue.type_hints[0] if len(clue.type_hints) == 1 else "unmarked"


def load_clues(posts: Iterable[RawPost]) -> list[ParsedClue]:
    """Every parsed clue, de-duplicated by key (a few posts repeat a clue)."""
    clues: dict[str, ParsedClue] = {}
    for post in posts:
        for clue in parse_clue_table(post):
            clues.setdefault(clue_key(clue), clue)
    return list(clues.values())


def sample_clues(clues: Sequence[ParsedClue], n: int, seed: int) -> list[ParsedClue]:
    """A reproducible sample spread evenly across clue kinds (round-robin by kind).

    Stratifying keeps rare kinds (reversals, homophones) from being drowned out
    by anagrams, so per-kind accuracy means something.
    """
    rng = random.Random(seed)
    by_kind: defaultdict[str, list[ParsedClue]] = defaultdict(list)
    for clue in sorted(clues, key=clue_key):  # sorted: same input order every time
        by_kind[kind_of(clue)].append(clue)
    for group in by_kind.values():
        rng.shuffle(group)
    kinds = sorted(by_kind)
    picked: list[ParsedClue] = []
    while len(picked) < n and any(by_kind.values()):
        for kind in kinds:
            if by_kind[kind] and len(picked) < n:
                picked.append(by_kind[kind].pop())
    return picked


def reveal_letters(answer: str, every: int) -> str:
    """Simulated crossing letters: show every `every`-th letter, 'MERINGUES' -> 'M?R?N?U?S'."""
    return "".join(ch if i % every == 0 else "?" for i, ch in enumerate(answer))


def to_row(clue: ParsedClue, result: SolveResult, seconds: float, pattern: str | None) -> ResultRow:
    status = result.verdict.status if result.verdict else "error"
    return ResultRow(
        key=clue_key(clue),
        clue=clue.clue_text,
        enumeration=clue.enumeration,
        kind=kind_of(clue),
        expected=clue.answer,
        answer=result.answer,
        status=status,
        right=result.answer == clue.answer,
        tokens=result.usage.total_tokens,
        tool_calls=sum(1 for s in result.steps if s.kind == "tool"),
        seconds=round(seconds, 1),
        pattern=pattern,
        error=result.error[:300],
    )


def run_evaluation(
    clues: Sequence[ParsedClue],
    make_solver: Callable[[Toolbox], Solver],
    base_toolbox: Toolbox,
    out_path: Path,
    *,
    reveal_every: int | None = None,
    on_row: Callable[[ResultRow, int, int], None] | None = None,
) -> list[ResultRow]:
    """Solve each clue not already in `out_path`, appending results as they come."""
    done = {r.key for r in read_jsonl(out_path, ResultRow)} if out_path.exists() else set()
    todo = [c for c in clues if clue_key(c) not in done]
    logger.info("%d clues: %d already done, %d to solve", len(clues), len(done), len(todo))

    for i, clue in enumerate(todo, start=1):
        toolbox = Toolbox(
            base_toolbox.dictionary, base_toolbox.lexicon, exclude_urls=[clue.source_url]
        )
        pattern = reveal_letters(clue.answer, reveal_every) if reveal_every else None
        started = time.monotonic()
        result = make_solver(toolbox).solve(clue.clue_text, clue.enumeration, pattern=pattern)
        row = to_row(clue, result, time.monotonic() - started, pattern)
        append_jsonl(out_path, [row])
        if on_row:
            on_row(row, i, len(todo))
    return [r for r in read_jsonl(out_path, ResultRow) if r.key in {clue_key(c) for c in clues}]


@dataclass(frozen=True)
class Summary:
    total: int
    right: int
    by_status: dict[str, tuple[int, int]]  # status -> (right, total)
    by_kind: dict[str, tuple[int, int]]
    confirmed_wrong: list[ResultRow]
    tokens: int


def summarise(rows: Sequence[ResultRow]) -> Summary:
    def tally(key: Callable[[ResultRow], str]) -> dict[str, tuple[int, int]]:
        counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            counts[key(row)][0] += row.right
            counts[key(row)][1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(counts.items())}

    return Summary(
        total=len(rows),
        right=sum(r.right for r in rows),
        by_status=tally(lambda r: r.status),
        by_kind=tally(lambda r: r.kind),
        confirmed_wrong=[r for r in rows if r.status == "confirmed" and not r.right],
        tokens=sum(r.tokens for r in rows),
    )


def format_report(summary: Summary) -> str:
    def pct(right: int, total: int) -> str:
        return f"{right:>3}/{total:<3} {right / total:6.1%}" if total else "  -"

    lines = [f"Overall   {pct(summary.right, summary.total)}", "", "By verdict:"]
    for status in ("confirmed", "pencilled", "unsure", "error"):
        if status in summary.by_status:
            lines.append(f"  {status:<10} {pct(*summary.by_status[status])}")
    lines += ["", "By clue type (blogger's label):"]
    lines += [f"  {kind:<20} {pct(*counts)}" for kind, counts in summary.by_kind.items()]
    lines += [
        "",
        f"CONFIRMED BUT WRONG: {len(summary.confirmed_wrong)}  (must stay 0)",
        *[
            f"  {r.clue} ({r.enumeration}): said {r.answer}, expected {r.expected}"
            for r in summary.confirmed_wrong
        ],
        "",
        f"Tokens: {summary.tokens:,} ({summary.tokens // max(summary.total, 1):,} per clue)",
    ]
    return "\n".join(lines)


def load_posts(path: Path) -> list[RawPost]:
    return read_jsonl(path, RawPost)
