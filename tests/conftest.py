"""Fixtures shared by several test files (pytest discovers this file automatically)."""

import sqlite3
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from cryptic_agent.lexicon.build import build_lexicon
from cryptic_agent.lexicon.sources import CRYPTICS, MOBY
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.lexicon.wordlist import UKACD_MEMBER

FAKE_UKACD = "\n".join(
    [
        "Copyright (c) 2009 J Ross Beresford",
        "All rights reserved.",
        "-" * 68,
        "a",
        "Aaron's rod",
        "eels",
        "no sweat",
        "pr�cis",  # an entry whose accent was lost upstream
        "",
        "senator",
        "treason",
        "Verdi",
    ]
)


@pytest.fixture
def fake_ukacd_zip(tmp_path: Path) -> Path:
    """A miniature copy of the UKACD archive, laid out like the real one."""
    path = tmp_path / "ccxxv-0.1.4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(UKACD_MEMBER, FAKE_UKACD)
    return path


# --- a miniature lexicon built from tiny copies of the real sources ---


GUARDIAN_1 = "https://fifteensquared.net/guardian-1/"
GUARDIAN_2 = "https://fifteensquared.net/guardian-2/"
TIMES_1 = "https://times.example/1/"

# rowid, clue, answer, definition, source_url
CLUES = [
    (1, "Love god's sparkling rose (4)", "EROS", "Love god", GUARDIAN_1),
    (2, "Love god in hero's tale (4)", "EROS", "Love god", GUARDIAN_2),
    (3, "Statue of love god (4)", "EROS", "love god", TIMES_1),
    (4, "Ruined a sculpture (4)", "WREN", "Ruined/a sculpture", TIMES_1),
    (5, "See 2 (5)", "PRECIS", "2", GUARDIAN_1),  # cross-reference: not a definition
    (6, "Summary of new car spice (6)", "PRÉCIS", "Summary", GUARDIAN_2),
    (7, "No answer recorded (5)", None, None, TIMES_1),
]
CHARADES = [  # clue_rowid, charade, answer
    (1, "sailor", "AB"),
    (2, "sailor", "AB"),
    (3, "sailor", "AB"),
    (4, "sailor", "SALT"),  # seen once: below the default min_support of 3
    (4, "a", "GR"),  # extraction noise, seen once
]
INDICATORS = [  # clue_rowid, anagram, container, insertion, reversal
    (1, "sparkling", "", "", ""),
    (2, "", "", "in", ""),
    (3, "", "about", "", "about"),
    (4, "ruined/broken", "", "", ""),
]
MOBY_LINES = [
    "joyful,festive,glad,merry",
    "summary,precis,digest",
]


def _write_cryptics_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE clues (rowid INTEGER PRIMARY KEY, clue TEXT, answer TEXT, definition TEXT,
            clue_number TEXT, puzzle_date TEXT, puzzle_name TEXT, source_url TEXT, source TEXT);
        CREATE TABLE charades_by_clue (clue_rowid INT, charade TEXT, answer TEXT);
        CREATE TABLE indicators_by_clue (clue_rowid INT PRIMARY KEY, alternation TEXT DEFAULT '',
            anagram TEXT DEFAULT '', container TEXT DEFAULT '', deletion TEXT DEFAULT '',
            hidden TEXT DEFAULT '', homophone TEXT DEFAULT '', insertion TEXT DEFAULT '',
            reversal TEXT DEFAULT '');
        """
    )
    con.executemany(
        "INSERT INTO clues (rowid, clue, answer, definition, source_url, source) "
        "VALUES (?, ?, ?, ?, ?, 'test')",
        CLUES,
    )
    con.executemany("INSERT INTO charades_by_clue VALUES (?, ?, ?)", CHARADES)
    con.executemany(
        "INSERT INTO indicators_by_clue (clue_rowid, anagram, container, insertion, reversal) "
        "VALUES (?, ?, ?, ?, ?)",
        INDICATORS,
    )
    con.commit()
    con.close()


@pytest.fixture
def guardian_urls() -> list[str]:
    """Two puzzles that both contain a "Love god -> EROS" clue."""
    return [GUARDIAN_1, GUARDIAN_2]


@pytest.fixture
def lexicon_path(tmp_path: Path, fake_ukacd_zip: Path) -> Path:
    # fake_ukacd_zip already lives in tmp_path under UKACD's real filename.
    (tmp_path / MOBY.filename).write_text("\r\n".join(MOBY_LINES), encoding="latin-1")
    _write_cryptics_db(tmp_path / CRYPTICS.filename)
    return build_lexicon(tmp_path, tmp_path / "lexicon.sqlite").path


@pytest.fixture
def lexicon(lexicon_path: Path) -> Iterator[Lexicon]:
    lx = Lexicon(lexicon_path)
    yield lx
    lx.close()
