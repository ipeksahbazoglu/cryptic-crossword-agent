from typing import Any

import pytest
from pydantic import ValidationError

from cryptic_agent.models import ClueRecord, enumeration_lengths, normalize_answer


def make_clue(**overrides: Any) -> ClueRecord:
    fields: dict[str, Any] = {
        "clue_text": "Senator arranged crime",
        "enumeration": "7",
        "answer": "TREASON",
        "definition": "crime",
        "wordplay_type": "anagram",
        "wordplay_explanation": "Anagram of SENATOR, indicated by 'arranged'",
        "source_url": "https://fifteensquared.net/example",
        "source_title": "Guardian 12,345",
    }
    fields.update(overrides)
    return ClueRecord.model_validate(fields)


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


def test_valid_clue_parses() -> None:
    clue = make_clue()

    assert clue.answer == "TREASON"
    assert clue.number is None


def test_answer_and_enumeration_are_normalized() -> None:
    clue = make_clue(enumeration="(3, 5)", answer="ice-cream")

    assert clue.enumeration == "3,5"
    assert clue.answer == "ICECREAM"


def test_answer_length_must_match_enumeration() -> None:
    with pytest.raises(ValidationError, match="needs 6"):
        make_clue(enumeration="6")


@pytest.mark.parametrize("enumeration", ["", "seven", "3,,4", "3/4"])
def test_malformed_enumeration_is_rejected(enumeration: str) -> None:
    with pytest.raises(ValidationError, match="not an enumeration"):
        make_clue(enumeration=enumeration)


def test_unknown_wordplay_type_is_rejected() -> None:
    with pytest.raises(ValidationError, match="wordplay_type"):
        make_clue(wordplay_type="spoonerism")


def test_records_are_immutable() -> None:
    clue = make_clue()

    with pytest.raises(ValidationError):
        clue.answer = "SENATOR"  # type: ignore[misc]
