"""Data shapes passed between pipeline stages.

Every record that crosses a stage boundary (disk, LLM output) is parsed into one
of these models, so the rest of the code can trust its fields instead of
re-checking dict keys.

Clue anatomy: every cryptic clue has a DEFINITION (a straight synonym at the
start or end) plus WORDPLAY, built from components that each have an INDICATOR
(signals the mechanism, adds no letters) and FODDER (what the mechanism works
on). "Senator arranged crime (7)" -> TREASON: definition "crime", indicator
"arranged" (anagram), fodder "Senator".
"""

import re
from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

# Mechanisms that build part of the answer. A charade ("D + A + CAPO") is one
# component; several components means a chained mechanism, e.g. an anagram
# placed inside a container.
ComponentType = Literal[
    "anagram",
    "charade",
    "hidden_word",
    "reversal",
    "homophone",
    "container",
    "deletion",
    "acrostic",
    "other",
]

# How a whole clue works, derived from its structure (see Clue.category).
ClueCategory = (
    ComponentType | Literal["double_definition", "cryptic_definition", "and_lit", "compound"]
)

DefinitionPosition = Literal["start", "end", "whole"]

_ENUMERATION_RE = re.compile(r"^\d+(?:[,-]\d+)*$")


def enumeration_lengths(enumeration: str) -> list[int]:
    """Word lengths in an enumeration: '7' -> [7], '3,4' -> [3, 4], '5-3' -> [5, 3]."""
    return [int(part) for part in re.split(r"[,-]", enumeration)]


def normalize_answer(answer: str) -> str:
    """Uppercase letters only: 'ice-cream' -> 'ICECREAM'."""
    return re.sub(r"[^A-Z]", "", answer.upper())


def normalize_phrase(text: str) -> str:
    """Lowercase words separated by single spaces: 'Senator, arranged!' -> 'senator arranged'.

    Used to compare clue fragments while ignoring case and punctuation. Apostrophes
    are dropped rather than split on, so "setter's" stays one word.
    """
    text = text.lower().replace("'", "").replace("’", "")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _contains_phrase(clue: str, phrase: str) -> bool:
    """Whole-word match: 'in' is found in 'bird in hand' but not in 'senator'."""
    return f" {normalize_phrase(phrase)} " in f" {normalize_phrase(clue)} "


class RawPost(BaseModel):
    """One Fifteensquared blog post, as written by the scraper."""

    model_config = ConfigDict(frozen=True)

    id: int
    date: datetime
    url: str
    title: str
    content_markdown: str


class Definition(BaseModel):
    """The straight-synonym part of a clue."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(..., min_length=1, description="The definition words as they appear")
    position: DefinitionPosition = Field(
        ..., description="'start' or 'end' of the clue, or 'whole' for &lit/cryptic definitions"
    )


class WordplayComponent(BaseModel):
    """One mechanism in the clue's wordplay."""

    model_config = ConfigDict(frozen=True)

    wordplay_type: ComponentType
    indicator_text: str | None = Field(
        None, description="Clue words signalling the mechanism; None when there is none"
    )
    fodder_text: str = Field(..., min_length=1, description="Clue words the mechanism acts on")
    explanation: str = Field(..., min_length=1, description="How this component builds letters")


class Clue(BaseModel):
    """A solved clue broken into its definition(s) and wordplay component(s)."""

    model_config = ConfigDict(frozen=True)

    clue_text: str = Field(..., min_length=1, description="The clue as printed, no enumeration")
    enumeration: str = Field(..., description="Word lengths, e.g. '7', '3,4' or '5-3'")
    answer: str = Field(..., description="The solution, uppercase, no spaces/punctuation")
    definitions: list[Definition] = Field(..., min_length=1)
    wordplay: list[WordplayComponent] = Field(default_factory=list)
    source_url: str
    source_title: str
    setter: str | None = None
    puzzle_date: date | None = None
    number: int | None = Field(None, description="Clue number in the grid, e.g. 7")
    direction: Literal["across", "down"] | None = None

    @field_validator("enumeration", mode="before")
    @classmethod
    def _clean_enumeration(cls, value: object) -> object:
        # Accept "(3, 4)" as well as "3,4"; anything else is rejected.
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

    @model_validator(mode="after")
    def _definitions_sit_where_they_claim(self) -> Self:
        clue = normalize_phrase(self.clue_text)
        for d in self.definitions:
            text = normalize_phrase(d.text)
            fits = {
                "start": clue.startswith(text + " ") or clue == text,
                "end": clue.endswith(" " + text) or clue == text,
                "whole": clue == text,
            }[d.position]
            if not fits:
                raise ValueError(f"definition {d.text!r} is not at the {d.position} of the clue")
        return self

    @model_validator(mode="after")
    def _components_come_from_the_clue(self) -> Self:
        for c in self.wordplay:
            if not _contains_phrase(self.clue_text, c.fodder_text):
                raise ValueError(f"fodder {c.fodder_text!r} does not appear in the clue")
            if c.indicator_text is not None and not _contains_phrase(
                self.clue_text, c.indicator_text
            ):
                raise ValueError(f"indicator {c.indicator_text!r} does not appear in the clue")
        return self

    @model_validator(mode="after")
    def _structure_is_a_known_clue_shape(self) -> Self:
        positions = sorted(d.position for d in self.definitions)
        if len(self.definitions) > 1:
            if self.wordplay:
                raise ValueError("a multiple-definition clue has no wordplay components")
            if "whole" in positions:
                raise ValueError("a double definition cannot include a whole-clue definition")
        elif positions != ["whole"] and not self.wordplay:
            raise ValueError("a clue with a start/end definition needs wordplay components")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def category(self) -> ClueCategory:
        """How the clue works overall; used to break down eval results."""
        if len(self.definitions) > 1:
            return "double_definition"
        if self.definitions[0].position == "whole":
            return "and_lit" if self.wordplay else "cryptic_definition"
        if len(self.wordplay) > 1:
            return "compound"
        return self.wordplay[0].wordplay_type
