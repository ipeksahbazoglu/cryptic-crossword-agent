"""Build data/lexicon/lexicon.sqlite from the downloaded sources.

One local SQLite file holds everything the solver looks up while it thinks:

    definitions   past definition phrase -> answer        (cryptics dataset)
    abbreviations clue phrase -> letters, e.g. sailor -> AB (cryptics dataset + curated list)
    indicators    indicator phrase -> wordplay type        (cryptics dataset)
    thesaurus     word or phrase -> related term           (Moby, both directions)
    clue_refs     the past clue each mined row came from  (for citing, and for
                  excluding test puzzles during evaluation)
    words         extra valid words: well-attested past answers, and UKACD
                  entries whose lost accents another source confirmed

Phrases are stored normalised (normalize_phrase) and answers as letters only
(normalize_answer), so lookups ignore case, punctuation and accents.
"""

import logging
import os
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from cryptic_agent.lexicon.sources import CRYPTICS, MOBY, UKACD
from cryptic_agent.lexicon.wordlist import damaged_pattern, read_ukacd
from cryptic_agent.models import normalize_answer, normalize_phrase, strip_accents

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE clue_refs (
    id INTEGER PRIMARY KEY,  -- rowid in the cryptics dataset
    source TEXT, source_url TEXT, puzzle_name TEXT, puzzle_date TEXT,
    clue TEXT, answer TEXT
);
CREATE TABLE definitions (
    phrase TEXT NOT NULL, answer TEXT NOT NULL, length INTEGER NOT NULL,
    clue_ref INTEGER REFERENCES clue_refs(id)
);
CREATE TABLE abbreviations (
    phrase TEXT NOT NULL, letters TEXT NOT NULL,
    clue_ref INTEGER REFERENCES clue_refs(id),  -- NULL for the curated list
    curated INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE indicators (
    phrase TEXT NOT NULL, wordplay TEXT NOT NULL,
    clue_ref INTEGER REFERENCES clue_refs(id)
);
CREATE TABLE thesaurus (
    phrase TEXT NOT NULL, term TEXT NOT NULL, length INTEGER NOT NULL,
    PRIMARY KEY (phrase, term)
) WITHOUT ROWID;
CREATE TABLE words (word TEXT PRIMARY KEY, origin TEXT NOT NULL) WITHOUT ROWID;
"""

INDEXES = """
CREATE INDEX definitions_phrase ON definitions (phrase, length);
CREATE INDEX abbreviations_phrase ON abbreviations (phrase);
CREATE INDEX indicators_phrase ON indicators (phrase);
CREATE INDEX clue_refs_url ON clue_refs (source_url);
"""

INDICATOR_COLUMNS = (
    "alternation",
    "anagram",
    "container",
    "deletion",
    "hidden",
    "homophone",
    "insertion",
    "reversal",
)
MIN_ANSWER_SUPPORT = 2  # a past answer becomes a dictionary word only if seen in 2+ clues


@dataclass(frozen=True)
class BuildSummary:
    path: Path
    counts: dict[str, int]


def _clean_definitions(raw: str | None) -> Iterator[str]:
    """Split 'Ruined/a sculpture' into parts; drop cross-references like '2' or '9'."""
    for part in (raw or "").split("/"):
        phrase = normalize_phrase(part)
        if phrase and phrase != "none" and not re.fullmatch(r"[\d ]+", phrase):
            yield phrase


def _import_cryptics(con: sqlite3.Connection, cryptics_db: Path) -> None:
    con.execute("ATTACH DATABASE ? AS src", (str(cryptics_db),))  # only ever SELECTed from
    con.execute(
        "INSERT INTO clue_refs SELECT rowid, source, source_url, puzzle_name, puzzle_date, "
        "clue, answer FROM src.clues"
    )

    def definition_rows() -> Iterator[tuple[str, str, int, int]]:
        for rowid, answer, definition in con.execute(
            "SELECT rowid, answer, definition FROM src.clues"
        ).fetchall():
            letters = normalize_answer(answer or "")
            if letters:
                for phrase in _clean_definitions(definition):
                    yield phrase, letters, len(letters), rowid

    con.executemany("INSERT INTO definitions VALUES (?, ?, ?, ?)", definition_rows())

    def abbreviation_rows() -> Iterator[tuple[str, str, int]]:
        for rowid, charade, answer in con.execute(
            "SELECT clue_rowid, charade, answer FROM src.charades_by_clue"
        ).fetchall():
            phrase, letters = normalize_phrase(charade or ""), normalize_answer(answer or "")
            if phrase and letters:
                yield phrase, letters, rowid

    con.executemany(
        "INSERT INTO abbreviations (phrase, letters, clue_ref) VALUES (?, ?, ?)",
        abbreviation_rows(),
    )

    def indicator_rows() -> Iterator[tuple[str, str, int]]:
        columns = ", ".join(INDICATOR_COLUMNS)
        for rowid, *cells in con.execute(
            f"SELECT clue_rowid, {columns} FROM src.indicators_by_clue"
        ).fetchall():
            for wordplay, cell in zip(INDICATOR_COLUMNS, cells, strict=True):
                for part in (cell or "").split("/"):
                    if phrase := normalize_phrase(part):
                        yield phrase, wordplay, rowid

    con.executemany("INSERT INTO indicators VALUES (?, ?, ?)", indicator_rows())
    con.commit()
    con.execute("DETACH DATABASE src")


def read_curated_abbreviations() -> Iterator[tuple[str, str]]:
    text = (
        resources.files("cryptic_agent.lexicon")
        .joinpath("data/abbreviations.tsv")
        .read_text(encoding="utf-8")
    )
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            phrase, letters = line.split("\t")
            for option in letters.split(","):
                yield normalize_phrase(phrase), normalize_answer(option)


def read_moby(path: Path) -> Iterator[tuple[str, list[str]]]:
    """(headword, related terms) per line. Moby is ASCII with accents already stripped."""
    with path.open(encoding="latin-1") as f:
        for line in f:
            root, *terms = line.rstrip("\r\n").split(",")
            yield root, terms


def _thesaurus_rows(moby: Path) -> Iterator[tuple[str, str, int]]:
    """Both directions: 'joyful' lists 'festive', so 'festive' also leads to JOYFUL."""
    for root, terms in read_moby(moby):
        root_phrase, root_letters = normalize_phrase(root), normalize_answer(root)
        for term in terms:
            term_phrase, term_letters = normalize_phrase(term), normalize_answer(term)
            if root_phrase and term_letters:
                yield root_phrase, term_letters, len(term_letters)
            if term_phrase and root_letters:
                yield term_phrase, root_letters, len(root_letters)


def repair_damaged(damaged: Iterable[str], attested: Iterable[str]) -> dict[str, str]:
    """Restore UKACD entries that lost accents, using only words another source contains.

    'pr\\ufffdcis' becomes 'precis' only if exactly one attested word fits the pattern;
    ambiguous ('caf\\ufffd' fits both 'cafe' and 'caff') or unmatched entries stay out.
    """
    by_length: defaultdict[int, set[str]] = defaultdict(set)
    for word in attested:
        plain = strip_accents(word).lower()
        by_length[len(plain)].add(plain)

    repaired: dict[str, str] = {}
    for entry in damaged:
        pattern = damaged_pattern(entry)
        fits = [w for w in by_length[len(entry)] if pattern.fullmatch(w)]
        if len(fits) == 1:
            repaired[entry] = fits[0]
    return repaired


def _attested_words(con: sqlite3.Connection, moby: Path) -> set[str]:
    """Words other sources vouch for: Moby headwords/terms and past answers, as printed."""
    words: set[str] = set()
    for root, terms in read_moby(moby):
        words.add(root.strip())
        words.update(t.strip() for t in terms)
    words.update(
        answer.lower()
        for (answer,) in con.execute("SELECT answer FROM clue_refs WHERE answer IS NOT NULL")
    )
    return {w for w in words if w}


def build_lexicon(sources_dir: Path, out_path: Path) -> BuildSummary:
    """Build the lexicon from the downloaded sources, replacing `out_path` atomically."""
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    moby = sources_dir / MOBY.filename
    con = sqlite3.connect(tmp_path)
    try:
        con.executescript(SCHEMA)

        logger.info("importing the cryptics dataset")
        _import_cryptics(con, sources_dir / CRYPTICS.filename)

        logger.info("adding curated abbreviations")
        con.executemany(
            "INSERT INTO abbreviations (phrase, letters, curated) VALUES (?, ?, 1)",
            read_curated_abbreviations(),
        )

        logger.info("importing the Moby thesaurus")
        con.executemany("INSERT OR IGNORE INTO thesaurus VALUES (?, ?, ?)", _thesaurus_rows(moby))

        logger.info("adding well-attested past answers and repaired UKACD entries")
        answer_support: defaultdict[str, int] = defaultdict(int)
        for (answer,) in con.execute("SELECT answer FROM clue_refs"):
            if letters := normalize_answer(answer or ""):
                answer_support[letters] += 1  # one count per distinct past clue
        con.executemany(
            "INSERT OR IGNORE INTO words VALUES (?, 'past answer')",
            ((w,) for w, n in answer_support.items() if n >= MIN_ANSWER_SUPPORT),
        )
        ukacd = read_ukacd(sources_dir / UKACD.filename)
        repaired = repair_damaged(ukacd.damaged, _attested_words(con, moby))
        con.executemany(
            "INSERT OR IGNORE INTO words VALUES (?, 'repaired UKACD')",
            ((normalize_answer(w),) for w in repaired.values()),
        )

        logger.info("indexing")
        con.executescript(INDEXES)
        con.commit()
        counts = {
            table: con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("clue_refs", "definitions", "abbreviations", "indicators", "thesaurus")
        }
        counts["words"] = con.execute("SELECT COUNT(*) FROM words").fetchone()[0]
        counts["ukacd_repaired"] = len(repaired)
        counts["ukacd_damaged"] = len(ukacd.damaged)
    finally:
        con.close()
    os.replace(tmp_path, out_path)
    return BuildSummary(path=out_path, counts=counts)
