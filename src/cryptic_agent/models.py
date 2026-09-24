"""Data shapes passed between pipeline stages.

Every record that crosses a stage boundary (disk, LLM output) is parsed into one
of these models, so the rest of the code can trust its fields instead of
re-checking dict keys.
"""

import re
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WordplayType = Literal[
    "anagram",
    "charade",
    "hidden_word",
    "reversal",
    "homophone",
    "double_definition",
    "container",
    "deletion",
    "acrostic",
    "and_lit",
    "cryptic_definition",
    "other",
]

_ENUMERATION_RE = re.compile(r"^\d+(?:[,-]\d+)*$")


def enumeration_lengths(enumeration: str) -> list[int]:
    """Word lengths in an enumeration: '7' -> [7], '3,4' -> [3, 4], '5-3' -> [5, 3]."""
    return [int(part) for part in re.split(r"[,-]", enumeration)]


def normalize_answer(answer: str) -> str:
    """Uppercase letters only: 'ice-cream' -> 'ICECREAM'."""
    return re.sub(r"[^A-Z]", "", answer.upper())


class RawPost(BaseModel):
    """One Fifteensquared blog post, as written by the scraper."""

    model_config = ConfigDict(frozen=True)

    id: int
    date: datetime
    url: str
    title: str
    content_markdown: str


class ClueRecord(BaseModel):
    """One clue with its answer and an explanation of the wordplay."""

    model_config = ConfigDict(frozen=True)

    clue_text: str = Field(..., min_length=1, description="The clue as printed, no enumeration")
    enumeration: str = Field(..., description="Word lengths, e.g. '7', '3,4' or '5-3'")
    answer: str = Field(..., description="The solution, uppercase, no spaces/punctuation")
    definition: str = Field(..., min_length=1, description="The definition part of the clue")
    wordplay_type: WordplayType
    wordplay_explanation: str = Field(
        ..., description="Plain-language explanation of how the wordplay produces the answer"
    )
    number: int | None = Field(None, description="Clue number in the grid, e.g. 7")
    direction: Literal["across", "down"] | None = None
    setter: str | None = None
    source_url: str
    source_title: str

    @field_validator("enumeration", mode="before")
    @classmethod
    def _clean_enumeration(cls, value: object) -> object:
        # Accept "(3, 4)" as well as "3,4"; anything else is rejected below.
        if isinstance(value, str):
            value = re.sub(r"[()\s]", "", value)
            if not _ENUMERATION_RE.fullmatch(value):
                raise ValueError(f"not an enumeration like '7' or '3,4': {value!r}")
        return value

    @field_validator("answer", mode="before")
    @classmethod
    def _clean_answer(cls, value: object) -> object:
        return normalize_answer(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _answer_fits_enumeration(self) -> Self:
        expected = sum(enumeration_lengths(self.enumeration))
        if len(self.answer) != expected:
            raise ValueError(
                f"answer {self.answer!r} has {len(self.answer)} letters "
                f"but enumeration ({self.enumeration}) needs {expected}"
            )
        return self
