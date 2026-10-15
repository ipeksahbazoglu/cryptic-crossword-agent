"""The fast pass on the miniature lexicon (tests/conftest.py): Love god -> EROS in
3 past clues, and "sparkling" known as an anagram indicator."""

import pytest

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

DICTIONARY = Dictionary(
    ["eros", "ores", "roes", "sore", "rose", "scrum", "crowd", "stand-in", "treason", "senator"]
)


def answers(result: FastPass) -> list[str]:
    return [c.answer for c in result.candidates]


def test_definition_and_wordplay_meeting_is_strong(lexicon: Lexicon) -> None:
    result = fast_pass("Love god's sparkling rose", "4", DICTIONARY, lexicon)

    top = result.candidates[0]
    assert top.answer == "EROS" and top.strong
    assert top.definitions[0].text == "Love god"  # the trailing 's read as "is"
    assert {w.indicator for w in top.wordplay if w.mechanism == "anagram"} == {"sparkling"}
    assert "EROS [STRONG]" in result.summary()


def test_anagram_never_returns_the_fodder_itself(lexicon: Lexicon) -> None:
    result = fast_pass("Love god's sparkling rose", "4", DICTIONARY, lexicon)

    assert "ROSE" not in answers(result)


def test_crossing_letters_filter_candidates(lexicon: Lexicon) -> None:
    result = fast_pass("Love god's sparkling rose", "4", DICTIONARY, lexicon, pattern="O???")

    assert answers(result) == ["ORES"]


def test_hidden_words_must_cross_a_word_boundary(lexicon: Lexicon) -> None:
    result = fast_pass("Serving of delicious crumble for crowd", "5", DICTIONARY, lexicon)

    hidden = {
        c.answer for c in result.candidates for w in c.wordplay if w.mechanism == "hidden_word"
    }
    assert "SCRUM" in hidden  # deliciouS CRUMble
    assert "CROWD" not in hidden  # just a word of the clue, not hidden


def test_multi_word_answers_must_be_whole_entries(lexicon: Lexicon) -> None:
    clue = "Locum openings for some trainee and new doctors in Norfolk"

    result = fast_pass(clue, "5-2", DICTIONARY, lexicon)

    assert answers(result) == ["STANDIN"]  # acrostic; no LOCUM+OP-style accidents


def test_excluded_puzzles_weaken_definition_evidence(
    lexicon: Lexicon, guardian_urls: list[str]
) -> None:
    result = fast_pass(
        "Love god's sparkling rose", "4", DICTIONARY, lexicon, exclude_urls=guardian_urls
    )

    eros = next(c for c in result.candidates if c.answer == "EROS")
    assert all("(3)" not in d.source for d in eros.definitions)


@pytest.mark.parametrize("clue", ["", "x"])
def test_degenerate_clues_do_not_crash(lexicon: Lexicon, clue: str) -> None:
    result = fast_pass(clue, "4", DICTIONARY, lexicon)

    assert "no quick wins" in result.summary() or result.candidates


def test_strong_needs_separate_words_for_definition_and_wordplay() -> None:
    # One word can't be both the definition and the fodder.
    definition = DefinitionHit("EROS", "rose", Span(3, 4), "thesaurus")
    same_words = WordplayHit("EROS", "anagram", "rose", Span(3, 4), indicator=None)
    other_words = WordplayHit("EROS", "anagram", "sore", Span(1, 2), indicator=None)

    assert not Candidate("EROS", [definition], [same_words]).strong
    assert Candidate("EROS", [definition], [other_words]).strong
