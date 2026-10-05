"""Read the UK Advanced Cryptics Dictionary (UKACD) word list.

UKACD is built for British cryptic crosswords: about 250k entries, including
British spellings, proper nouns (Verdi, Serbia) and phrases (no sweat). On our
scraped Quick Cryptic answers it accepts 99% versus 81% for NLTK's word list.

Known damage in the only copy still online: about 1,300 entries lost their
accented letters to an encoding conversion, so 'précis' is stored as 'pr\\ufffdcis'
(U+FFFD is the Unicode replacement character). The original letter can't be
recovered from this file, and guessing would invent words like PRACIS, so those
entries are kept apart as `damaged`; they are only restored when another source
contains a word that fits the pattern.
"""

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

UKACD_MEMBER = "ccxxv-0.1.4/ccxxv/wordlists/UKACD.txt"
_HEADER_END = "-" * 68  # the licence header ends with a line of dashes
REPLACEMENT_CHAR = "�"


@dataclass(frozen=True)
class Ukacd:
    licence: str
    entries: list[str]  # as printed: "Aaron's rod", "no sweat", "Verdi"
    damaged: list[str]  # entries containing REPLACEMENT_CHAR, e.g. "pr�cis"


def read_ukacd(zip_path: Path) -> Ukacd:
    with zipfile.ZipFile(zip_path) as archive:
        text = archive.read(UKACD_MEMBER).decode("utf-8")
    licence, _, body = text.partition(_HEADER_END)
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return Ukacd(
        licence=licence.strip(),
        entries=[line for line in lines if REPLACEMENT_CHAR not in line],
        damaged=[line for line in lines if REPLACEMENT_CHAR in line],
    )


def damaged_pattern(entry: str) -> re.Pattern[str]:
    """'pr\\ufffdcis' -> a pattern matching 'precis': each lost letter is one unknown letter."""
    parts = (re.escape(part) for part in entry.lower().split(REPLACEMENT_CHAR))
    return re.compile("[a-z]".join(parts))
