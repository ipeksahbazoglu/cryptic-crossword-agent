"""Shared text helpers and the scraped-post record.

    normalize_answer   'ice-cream' -> 'ICECREAM' (letters only, accents folded)
    normalize_phrase   'Senator, arranged!' -> 'senator arranged'
    enumeration_lengths '(3, 4)' -> [3, 4]
    RawPost            one Fifteensquared blog post, as written by the scraper

Parsed clues live in extraction/table_parser.py (ParsedClue) and the solver's
answer in agent/worksheet.py (Worksheet).
"""

import re
import unicodedata
from datetime import datetime

from pydantic import BaseModel, ConfigDict


def enumeration_lengths(enumeration: str) -> list[int]:
    """Word lengths in an enumeration: '7' -> [7], '(3, 4)' -> [3, 4], '5-3' -> [5, 3].

    Brackets and spaces are ignored, since models and people often include them.
    """
    cleaned = re.sub(r"[()\s]", "", enumeration)
    return [int(part) for part in re.split(r"[,\-\u2013]", cleaned)]


# Letters NFKD does not split into base letter + accent mark.
_UNDECOMPOSABLE = str.maketrans(
    {"æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "ß": "ss", "ł": "l", "Ł": "L"}
)


def strip_accents(text: str) -> str:
    """Plain letters: 'précis' -> 'precis', 'Rösti' -> 'Rosti', 'Ærø' -> 'AEro'.

    NFKD splits 'é' into 'e' plus a combining accent mark, which is then dropped.
    """
    decomposed = unicodedata.normalize("NFKD", text.translate(_UNDECOMPOSABLE))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_answer(answer: str) -> str:
    """Uppercase letters only: 'ice-cream' -> 'ICECREAM', 'précis' -> 'PRECIS'."""
    return re.sub(r"[^A-Z]", "", strip_accents(answer).upper())


def normalize_phrase(text: str) -> str:
    """Lowercase words separated by single spaces: 'Senator, arranged!' -> 'senator arranged'.

    Used to compare clue fragments while ignoring case and punctuation. Apostrophes
    are dropped rather than split on, so "setter's" stays one word.
    """
    text = strip_accents(text).lower().replace("'", "").replace("’", "")
    return " ".join(re.findall(r"[a-z0-9]+", text))


class RawPost(BaseModel):
    """One Fifteensquared blog post, as written by the scraper."""

    model_config = ConfigDict(frozen=True)

    id: int
    date: datetime
    url: str
    title: str
    content_markdown: str
