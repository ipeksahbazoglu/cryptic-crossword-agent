"""When is an answer allowed to be CONFIRMED? These tests pin the rules down."""

from typing import Any

import pytest

from cryptic_agent.agent.verify import Verdict, matches_pattern, verify
from cryptic_agent.agent.worksheet import Worksheet
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.tools.dictionary import Dictionary

DICTIONARY = Dictionary(["treason", "senator", "scrum", "pinch", "evergreen", "stand in", "bar"])
CLUE = "Senator arranged crime"


def worksheet(**overrides: Any) -> Worksheet:
    fields: dict[str, Any] = {
        "answer": "TREASON",
        "definitions": [{"text": "crime", "position": "end"}],
        "wordplay": [
            {
                "mechanism": "anagram",
                "indicator": "arranged",
                "fodder": "Senator",
                "produces": "TREASON",
                "explanation": "SENATOR rearranged",
            }
        ],
        "link_words": [],
        "confidence": "confirmed",
        "alternatives": [],
    }
    fields.update(overrides)
    return Worksheet.model_validate(fields)


def run(ws: Worksheet, clue: str = CLUE, enumeration: str = "7", **kw: Any) -> Verdict:
    return verify(ws, clue, enumeration, DICTIONARY, **kw)


def failed_names(verdict: Verdict) -> list[str]:
    return [c.name for c in verdict.failed()]


def test_a_fully_checked_parse_is_confirmed() -> None:
    verdict = run(worksheet())

    assert verdict.status == "confirmed"
    assert verdict.failed() == []


# --- things that keep an answer from being confirmed ------------------------------


def test_wrong_letters_for_an_anagram() -> None:
    step = worksheet().wordplay[0].model_copy(update={"fodder": "arranged crime"})

    verdict = run(worksheet(wordplay=[step]))

    assert verdict.status == "pencilled"
    # "Senator" now has no role either, so that check fails too.
    assert "wordplay: anagram of 'arranged crime'" in failed_names(verdict)


def test_definition_not_where_claimed() -> None:
    verdict = run(worksheet(definitions=[{"text": "crime", "position": "start"}]))

    assert verdict.status == "pencilled"
    assert failed_names(verdict) == ["definition 'crime'"]


def test_unexplained_clue_word() -> None:
    verdict = run(worksheet(), clue="Senator arranged our crime")

    assert verdict.status == "pencilled"
    assert "unexplained: ['our']" in verdict.failed()[0].detail


def test_link_words_count_as_roles() -> None:
    verdict = run(worksheet(link_words=["our"]), clue="Senator arranged our crime")

    assert verdict.status == "confirmed"


def test_indicator_not_in_clue() -> None:
    step = worksheet().wordplay[0].model_copy(update={"indicator": "broken"})

    assert run(worksheet(wordplay=[step])).status == "pencilled"


def test_model_doubt_is_respected() -> None:
    assert run(worksheet(confidence="pencilled")).status == "pencilled"


def test_unknown_word_is_not_confirmed() -> None:
    step = worksheet().wordplay[0].model_copy(update={"produces": "ATONERS"})

    verdict = run(worksheet(answer="ATONERS", wordplay=[step]))

    assert verdict.status == "pencilled"  # letters fit, but not a dictionary word


def test_a_word_cannot_have_two_roles() -> None:
    step = worksheet().wordplay[0].model_copy(update={"fodder": "crime", "produces": "MERIC"})
    ws = worksheet(answer="MERIC", wordplay=[step], link_words=["Senator"])

    verdict = run(ws, enumeration="5")

    role_check = next(c for c in verdict.checks if c.name == "every word has a role")
    assert role_check.passed is False
    assert "used more often than it appears: ['crime']" in role_check.detail


def test_a_repeated_clue_word_needs_a_role_each_time() -> None:
    verdict = run(worksheet(link_words=["in"]), clue="Senator in arranged in crime")

    assert verdict.status == "pencilled"
    assert "unexplained: ['in']" in verdict.failed()[0].detail


def test_and_lit_may_use_every_word_twice() -> None:
    # In an &lit clue the whole clue is the definition and also the wordplay.
    ws = worksheet(definitions=[{"text": "Senator arranged crime", "position": "whole"}])
    verdict = run(ws)

    assert next(c for c in verdict.checks if c.name == "every word has a role").passed


# --- when the wordplay fits several words, the definition must pick this one -----------

ANAGRAMS = Dictionary(["eros", "ores", "roes", "sore", "rose"])


def eros_sheet(answer: str) -> Worksheet:
    return worksheet(
        answer=answer,
        definitions=[{"text": "Love god's", "position": "start"}],
        wordplay=[
            {
                "mechanism": "anagram",
                "indicator": "sparkling",
                "fodder": "rose",
                "produces": answer,
                "explanation": "ROSE rearranged",
            }
        ],
    )


@pytest.mark.parametrize(
    ("answer", "status"),
    [("EROS", "confirmed"), ("SORE", "pencilled"), ("ORES", "pencilled")],
)
def test_ambiguous_wordplay_needs_the_definition(
    lexicon: Lexicon, answer: str, status: str
) -> None:
    # The miniature lexicon knows "Love god" -> EROS (and reads the 's as "is").
    verdict = verify(
        eros_sheet(answer), "Love god's sparkling rose", "4", ANAGRAMS, lexicon=lexicon
    )

    assert verdict.status == status


def test_ambiguous_wordplay_without_a_lexicon_is_never_confirmed() -> None:
    verdict = verify(eros_sheet("EROS"), "Love god's sparkling rose", "4", ANAGRAMS)

    assert verdict.status == "pencilled"


def test_excluded_puzzles_cannot_supply_the_definition(
    lexicon: Lexicon, guardian_urls: list[str]
) -> None:
    # All three "Love god" clues hidden: nothing left to tie the definition to EROS.
    verdict = verify(
        eros_sheet("EROS"),
        "Love god's sparkling rose",
        "4",
        ANAGRAMS,
        lexicon=lexicon,
        exclude_urls=[*guardian_urls, "https://times.example/1/"],
    )

    assert verdict.status == "pencilled"


def test_unambiguous_wordplay_needs_no_lexicon() -> None:
    verdict = run(worksheet())  # SENATOR has one anagram in this dictionary

    assert verdict.status == "confirmed"
    assert "definition means the answer" not in [c.name for c in verdict.checks]


# --- things that make an answer unsure ------------------------------------------------


def test_wrong_length_is_unsure() -> None:
    assert run(worksheet(), enumeration="6").status == "unsure"


def test_crossing_letters_contradict_the_answer() -> None:
    assert run(worksheet(), pattern="X??????").status == "unsure"
    assert run(worksheet(), pattern="T?E?S?N").status == "confirmed"


# --- mechanisms code can't check yet stay pencilled -----------------------------------


def test_double_definition_is_pencilled() -> None:
    ws = worksheet(
        answer="PINCH",
        definitions=[
            {"text": "Steal", "position": "start"},
            # "say" belongs to this definition: a pinch of salt, for example
            {"text": "small amount of salt, say", "position": "end"},
        ],
        wordplay=[],
    )

    verdict = run(ws, clue="Steal small amount of salt, say", enumeration="5")

    assert verdict.status == "pencilled"
    assert verdict.failed() == []


def test_synonym_charade_is_pencilled() -> None:
    ws = worksheet(
        answer="EVERGREEN",
        definitions=[{"text": "never unpopular", "position": "end"}],
        wordplay=[
            {
                "mechanism": "charade",
                "indicator": None,
                "fodder": "Always eco-friendly",
                "produces": "EVERGREEN",
                "explanation": "EVER (always) + GREEN (eco-friendly)",
            }
        ],
        link_words=["and"],
    )

    verdict = run(ws, clue="Always eco-friendly and never unpopular", enumeration="9")

    assert verdict.status == "pencilled"
    assert [c.passed for c in verdict.checks if c.name.startswith("wordplay:")] == [None]


# --- mechanical checks ---------------------------------------------------------------------


def test_hidden_word_with_a_split_indicator() -> None:
    ws = worksheet(
        answer="SCRUM",
        definitions=[{"text": "crowd", "position": "end"}],
        wordplay=[
            {
                "mechanism": "hidden_word",
                "indicator": "Serving of ... for",
                "fodder": "delicious crumble",
                "produces": "SCRUM",
                "explanation": "hidden in deliciouS CRUMble",
            }
        ],
    )

    verdict = run(ws, clue="Serving of delicious crumble for crowd", enumeration="5")

    assert verdict.status == "confirmed"


def test_acrostic_with_hyphenated_enumeration() -> None:
    ws = worksheet(
        answer="STANDIN",
        definitions=[{"text": "Locum", "position": "start"}],
        wordplay=[
            {
                "mechanism": "acrostic",
                "indicator": "openings for",
                "fodder": "some trainee and new doctors in Norfolk",
                "produces": "STANDIN",
                "explanation": "first letters",
            }
        ],
    )

    verdict = run(
        ws,
        clue="Locum openings for some trainee and new doctors in Norfolk",
        enumeration="5-2",
    )

    assert verdict.status == "confirmed"


@pytest.mark.parametrize(
    ("answer", "pattern", "fits"),
    [
        ("MERINGUES", "M?R?N?U?S", True),
        ("MERINGUES", "m_r_n_u_s", True),
        ("MERINGUES", "M.R.N.U.S", True),
        ("MERINGUES", "X?R?N?U?S", False),
        ("MERINGUES", "M?R?N", False),  # wrong length
    ],
)
def test_matches_pattern(answer: str, pattern: str, fits: bool) -> None:
    assert matches_pattern(answer, pattern) is fits
