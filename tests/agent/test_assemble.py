"""Tier 0 confirms answers with no model at all, so each safety rule is pinned here.

Uses the miniature lexicon: "Love god" -> EROS (3 past clues), "sparkling" is a
known anagram indicator.
"""

from cryptic_agent.agent.assemble import solve_by_code
from cryptic_agent.agent.fastpass import (
    Candidate,
    DefinitionHit,
    FastPass,
    Span,
    WordplayHit,
    fast_pass,
)
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.tools.dictionary import Dictionary

DICTIONARY = Dictionary(["eros", "ores", "roes", "sore", "rose"])


def test_a_fully_explained_clue_is_solved_by_code(lexicon: Lexicon) -> None:
    found = fast_pass("Love god's sparkling rose", "4", DICTIONARY, lexicon)

    solved = solve_by_code(found, DICTIONARY)

    assert solved is not None
    sheet, verdict = solved
    assert verdict.status == "confirmed" and verdict.answer == "EROS"
    assert sheet.definitions[0].text == "Love god's"  # the clue's own words, for verify
    assert sheet.wordplay[0].indicator == "sparkling"


def test_link_words_are_allowed(lexicon: Lexicon) -> None:
    found = fast_pass("Love god's sparkling rose for the", "4", DICTIONARY, lexicon)

    solved = solve_by_code(found, DICTIONARY)

    assert solved is not None and solved[0].link_words == ["for", "the"]


def test_an_unexplained_word_stops_tier_0(lexicon: Lexicon) -> None:
    found = fast_pass("Love god's sparkling rose today", "4", DICTIONARY, lexicon)

    assert solve_by_code(found, DICTIONARY) is None  # "today" has no role code can name


def test_no_indicator_no_tier_0(lexicon: Lexicon) -> None:
    found = fast_pass("Love god's rose", "4", DICTIONARY, lexicon)

    assert found.strong  # definition and anagram still agree...
    assert solve_by_code(found, DICTIONARY) is None  # ...but nothing signals the anagram


def test_two_confirmable_answers_means_no_guess() -> None:
    def candidate(answer: str) -> Candidate:
        return Candidate(
            answer,
            [DefinitionHit(answer, "Love god's", Span(0, 2), "past clues (3)")],
            [WordplayHit(answer, "anagram", "rose", Span(3, 4), "sparkling", Span(2, 3))],
        )

    found = FastPass("Love god's sparkling rose", "4", [candidate("EROS"), candidate("SORE")])

    assert solve_by_code(found, DICTIONARY) is None
    assert solve_by_code(FastPass(found.clue, "4", [candidate("EROS")]), DICTIONARY) is not None


def test_crossing_letters_are_respected(lexicon: Lexicon) -> None:
    found = fast_pass("Love god's sparkling rose", "4", DICTIONARY, lexicon, pattern="O???")

    assert solve_by_code(found, DICTIONARY, pattern="O???") is None
