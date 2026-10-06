from pathlib import Path
from typing import Any

import pytest

from cryptic_agent.agent.solver import Solver, SolveResult
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.agent.verify import Verdict
from cryptic_agent.evaluation.run import (
    ResultRow,
    clue_key,
    format_report,
    reveal_letters,
    run_evaluation,
    sample_clues,
    summarise,
)
from cryptic_agent.extraction.table_parser import ParsedClue
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.llm.client import RateLimitedError, ScriptedLLM, Usage
from cryptic_agent.tools.dictionary import Dictionary


def clue(number: int, kind: str, answer: str = "TREASON", url: str = "https://p/1") -> ParsedClue:
    return ParsedClue(
        number=number,
        direction="across",
        clue_markup="",
        clue_text=f"clue {number}",
        enumeration=str(len(answer)),
        answer=answer,
        definitions=[],
        indicators=[],
        type_hints=[kind] if kind != "unmarked" else [],
        parsing="",
        source_url=url,
        source_title="t",
        setter=None,
    )


CLUES = [
    clue(i, kind) for i, kind in enumerate(["anagram"] * 10 + ["reversal"] * 2 + ["unmarked"] * 3)
]


# --- sampling ------------------------------------------------------------------------


def test_sample_is_reproducible() -> None:
    assert sample_clues(CLUES, 6, seed=1) == sample_clues(list(reversed(CLUES)), 6, seed=1)
    assert sample_clues(CLUES, 6, seed=1) != sample_clues(CLUES, 6, seed=2)


def test_sample_spreads_across_kinds() -> None:
    kinds = [c.type_hints[0] if c.type_hints else "unmarked" for c in sample_clues(CLUES, 6, 1)]

    assert kinds.count("reversal") == 2  # rare kind fully represented, not drowned out
    assert kinds.count("unmarked") == 2
    assert kinds.count("anagram") == 2


def test_sample_never_exceeds_what_exists() -> None:
    assert len(sample_clues(CLUES, 100, seed=1)) == len(CLUES)


def test_reveal_letters() -> None:
    assert reveal_letters("MERINGUES", 2) == "M?R?N?U?S"
    assert reveal_letters("TREASON", 3) == "T??A??N"


# --- running -------------------------------------------------------------------------


class FakeSolver(Solver):
    """Answers TREASON for everything and records what it was asked."""

    calls: list[dict[str, Any]] = []

    def solve(self, clue: str, enumeration: str, *, pattern: str | None = None) -> SolveResult:
        FakeSolver.calls.append(
            {"clue": clue, "pattern": pattern, "excluded": list(self.toolbox.exclude_urls)}
        )
        verdict = Verdict(answer="TREASON", status="confirmed", checks=[])
        return SolveResult(clue, enumeration, pattern, [], None, verdict, Usage(100, 50))


def make_toolbox(lexicon: Lexicon) -> Toolbox:
    return Toolbox(Dictionary(["treason"]), lexicon)


def test_run_records_rows_and_hides_each_clues_own_puzzle(lexicon: Lexicon, tmp_path: Path) -> None:
    FakeSolver.calls = []
    clues = [clue(1, "anagram", url="https://p/1"), clue(2, "anagram", "SENATOR", "https://p/2")]

    rows = run_evaluation(
        clues,
        lambda tb: FakeSolver(ScriptedLLM([]), tb),
        make_toolbox(lexicon),
        tmp_path / "r.jsonl",
    )

    assert [r.right for r in rows] == [True, False]
    assert [c["excluded"] for c in FakeSolver.calls] == [["https://p/1"], ["https://p/2"]]


def test_run_resumes_without_redoing_clues(lexicon: Lexicon, tmp_path: Path) -> None:
    FakeSolver.calls = []
    out = tmp_path / "r.jsonl"
    make = lambda tb: FakeSolver(ScriptedLLM([]), tb)  # noqa: E731

    run_evaluation(CLUES[:3], make, make_toolbox(lexicon), out)
    rows = run_evaluation(CLUES[:5], make, make_toolbox(lexicon), out)

    assert len(FakeSolver.calls) == 5  # 3 then only the 2 new ones
    assert {r.key for r in rows} == {clue_key(c) for c in CLUES[:5]}


class QuotaSolver(FakeSolver):
    """Solves `budget` clues, then runs out of quota."""

    budget = 0

    def solve(self, clue: str, enumeration: str, *, pattern: str | None = None) -> SolveResult:
        if QuotaSolver.budget == 0:
            raise RateLimitedError("tokens per day (TPD)")
        QuotaSolver.budget -= 1
        return super().solve(clue, enumeration, pattern=pattern)


def test_out_of_quota_stops_the_run_and_resume_continues(lexicon: Lexicon, tmp_path: Path) -> None:
    FakeSolver.calls = []
    out = tmp_path / "r.jsonl"
    make = lambda tb: QuotaSolver(ScriptedLLM([]), tb)  # noqa: E731

    QuotaSolver.budget = 2
    with pytest.raises(RateLimitedError):
        run_evaluation(CLUES[:5], make, make_toolbox(lexicon), out)
    assert len(out.read_text().splitlines()) == 2  # nothing written for the refused clue

    QuotaSolver.budget = 10  # the next day
    rows = run_evaluation(CLUES[:5], make, make_toolbox(lexicon), out)

    assert len(rows) == 5
    assert len(FakeSolver.calls) == 5  # each clue solved exactly once


def test_crossing_letters_are_simulated(lexicon: Lexicon, tmp_path: Path) -> None:
    FakeSolver.calls = []

    rows = run_evaluation(
        [clue(1, "anagram")],
        lambda tb: FakeSolver(ScriptedLLM([]), tb),
        make_toolbox(lexicon),
        tmp_path / "r.jsonl",
        reveal_every=2,
    )

    assert FakeSolver.calls[0]["pattern"] == "T?E?S?N"
    assert rows[0].pattern == "T?E?S?N"


# --- reporting -----------------------------------------------------------------------


def row(kind: str, status: str, right: bool) -> ResultRow:
    return ResultRow(
        key=f"{kind}{status}{right}",
        clue="c",
        enumeration="7",
        kind=kind,
        expected="TREASON",
        answer="TREASON" if right else "SENATOR",
        status=status,
        right=right,
        tokens=1000,
        tool_calls=2,
        seconds=1.0,
    )


def test_summary_counts_by_verdict_and_kind() -> None:
    summary = summarise(
        [
            row("anagram", "confirmed", True),
            row("anagram", "pencilled", False),
            row("charade", "pencilled", True),
        ]
    )

    assert (summary.right, summary.total) == (2, 3)
    assert summary.by_status == {"confirmed": (1, 1), "pencilled": (1, 2)}
    assert summary.by_kind == {"anagram": (1, 2), "charade": (1, 1)}
    assert summary.confirmed_wrong == []


def test_confirmed_but_wrong_is_reported_loudly() -> None:
    summary = summarise([row("anagram", "confirmed", False)])

    report = format_report(summary)

    assert len(summary.confirmed_wrong) == 1
    assert "CONFIRMED BUT WRONG: 1" in report
    assert "said SENATOR, expected TREASON" in report
