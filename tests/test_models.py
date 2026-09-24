from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cryptic_agent.jsonl import read_jsonl, write_jsonl
from cryptic_agent.models import (
    Clue,
    enumeration_lengths,
    normalize_answer,
    normalize_phrase,
)

SOURCE = {"source_url": "https://fifteensquared.net/example", "source_title": "Guardian 12,345"}


def anagram_clue(**overrides: Any) -> Clue:
    """Senator arranged crime (7) -> TREASON."""
    fields: dict[str, Any] = {
        "clue_text": "Senator arranged crime",
        "enumeration": "7",
        "answer": "TREASON",
        "definitions": [{"text": "crime", "position": "end"}],
        "wordplay": [
            {
                "wordplay_type": "anagram",
                "indicator_text": "arranged",
                "fodder_text": "Senator",
                "explanation": "SENATOR rearranged",
            }
        ],
        **SOURCE,
    }
    fields.update(overrides)
    return Clue.model_validate(fields)


# --- helpers -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("enumeration", "expected"),
    [("7", [7]), ("3,4", [3, 4]), ("5-3", [5, 3]), ("2,3,4", [2, 3, 4])],
)
def test_enumeration_lengths(enumeration: str, expected: list[int]) -> None:
    assert enumeration_lengths(enumeration) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("treason", "TREASON"), ("ice cream", "ICECREAM"), ("Jack-in-the-box", "JACKINTHEBOX")],
)
def test_normalize_answer(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Senator, arranged!", "senator arranged"),
        ("  Setter's   ruse ", "setters ruse"),
        ("Setter’s ruse", "setters ruse"),
    ],
)
def test_normalize_phrase(raw: str, expected: str) -> None:
    assert normalize_phrase(raw) == expected


# --- valid clue shapes and their categories -----------------------------------


def test_single_component_clue_takes_its_component_type() -> None:
    clue = anagram_clue()

    assert clue.category == "anagram"
    assert clue.answer == "TREASON"


def test_hidden_word() -> None:
    clue = Clue.model_validate(
        {
            "clue_text": "Love story found in a Roman cerebral",
            "enumeration": "7",
            "answer": "romance",
            "definitions": [{"text": "Love story", "position": "start"}],
            "wordplay": [
                {
                    "wordplay_type": "hidden_word",
                    "indicator_text": "found in",
                    "fodder_text": "a Roman cerebral",
                    "explanation": "hidden in aROMAN CErebral",
                }
            ],
            **SOURCE,
        }
    )

    assert clue.category == "hidden_word"


def test_double_definition_has_two_definitions_and_no_wordplay() -> None:
    clue = Clue.model_validate(
        {
            "clue_text": "Put up with a large animal",
            "enumeration": "4",
            "answer": "BEAR",
            "definitions": [
                {"text": "Put up with", "position": "start"},
                {"text": "large animal", "position": "end"},
            ],
            **SOURCE,
        }
    )

    assert clue.category == "double_definition"


def test_cryptic_definition_is_one_whole_clue_definition() -> None:
    clue = Clue.model_validate(
        {
            "clue_text": "Flower of London",
            "enumeration": "6",
            "answer": "THAMES",
            "definitions": [{"text": "Flower of London", "position": "whole"}],
            **SOURCE,
        }
    )

    assert clue.category == "cryptic_definition"


def test_and_lit_is_whole_clue_definition_plus_wordplay() -> None:
    clue = anagram_clue(
        clue_text="Senator arranged this?",
        definitions=[{"text": "Senator arranged this?", "position": "whole"}],
    )

    assert clue.category == "and_lit"


def test_several_components_is_compound() -> None:
    clue = anagram_clue(
        clue_text="Senator arranged around the crime",
        wordplay=[
            {
                "wordplay_type": "anagram",
                "indicator_text": "arranged",
                "fodder_text": "Senator",
                "explanation": "SENATOR rearranged",
            },
            {
                "wordplay_type": "container",
                "indicator_text": "around",
                "fodder_text": "the",
                "explanation": "illustrative second component",
            },
        ],
    )

    assert clue.category == "compound"


def test_indicator_is_optional() -> None:
    clue = anagram_clue(
        wordplay=[
            {
                "wordplay_type": "charade",
                "fodder_text": "Senator arranged",
                "explanation": "charade without an indicator",
            }
        ]
    )

    assert clue.wordplay[0].indicator_text is None


# --- rejected clues --------------------------------------------------------------


def test_answer_and_enumeration_are_normalized() -> None:
    clue = Clue.model_validate(
        {
            "clue_text": "Frozen dessert",
            "enumeration": "(3, 5)",
            "answer": "ice-cream",
            "definitions": [{"text": "Frozen dessert", "position": "whole"}],
            **SOURCE,
        }
    )

    assert (clue.enumeration, clue.answer) == ("3,5", "ICECREAM")


def test_answer_length_must_match_enumeration() -> None:
    with pytest.raises(ValidationError, match="needs 6"):
        anagram_clue(enumeration="6")


@pytest.mark.parametrize("enumeration", ["", "seven", "3,,4", "3/4"])
def test_malformed_enumeration_is_rejected(enumeration: str) -> None:
    with pytest.raises(ValidationError, match="not an enumeration"):
        anagram_clue(enumeration=enumeration)


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        ({"text": "crime", "position": "start"}, "not at the start"),
        ({"text": "arranged", "position": "end"}, "not at the end"),
        ({"text": "crime", "position": "whole"}, "not at the whole"),
        ({"text": "rime", "position": "end"}, "not at the end"),  # partial word
    ],
)
def test_definition_must_sit_where_it_claims(definition: dict[str, str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        anagram_clue(definitions=[definition])


def test_fodder_must_appear_in_clue() -> None:
    component = {
        "wordplay_type": "anagram",
        "indicator_text": "arranged",
        "fodder_text": "Senators",
        "explanation": "x",
    }

    with pytest.raises(ValidationError, match="fodder 'Senators'"):
        anagram_clue(wordplay=[component])


def test_indicator_must_be_whole_clue_words() -> None:
    # "nat" is inside "Senator" but is not a word of the clue.
    component = {
        "wordplay_type": "anagram",
        "indicator_text": "nat",
        "fodder_text": "Senator",
        "explanation": "x",
    }

    with pytest.raises(ValidationError, match="indicator 'nat'"):
        anagram_clue(wordplay=[component])


def test_double_definition_cannot_have_wordplay() -> None:
    with pytest.raises(ValidationError, match="no wordplay components"):
        anagram_clue(
            definitions=[
                {"text": "Senator", "position": "start"},
                {"text": "crime", "position": "end"},
            ]
        )


def test_start_or_end_definition_needs_wordplay() -> None:
    with pytest.raises(ValidationError, match="needs wordplay components"):
        anagram_clue(wordplay=[])


def test_unknown_component_type_is_rejected() -> None:
    component = {
        "wordplay_type": "double_definition",  # a clue shape, not a component
        "fodder_text": "Senator",
        "explanation": "x",
    }

    with pytest.raises(ValidationError, match="wordplay_type"):
        anagram_clue(wordplay=[component])


def test_clues_are_immutable() -> None:
    clue = anagram_clue()

    with pytest.raises(ValidationError):
        clue.answer = "SENATOR"  # type: ignore[misc]


# --- persistence ----------------------------------------------------------------------


def test_round_trip_through_jsonl_keeps_category(tmp_path: Path) -> None:
    path = tmp_path / "clues.jsonl"
    clue = anagram_clue()

    write_jsonl(path, [clue])

    assert '"category":"anagram"' in path.read_text()  # handy for queries and eval
    assert read_jsonl(path, Clue) == [clue]
