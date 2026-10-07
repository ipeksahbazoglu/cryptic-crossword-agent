"""Look things up in the built lexicon (see build.py), the way a solver draws on experience.

Each method answers one question a human asks while solving, and returns the
evidence with how many past clues back it, so the caller can tell a well-worn
convention from a one-off:

    definition_answers("sweet desserts", length=9)  what did this definition lead to before?
    synonyms("joyful", length=7)                    what does the thesaurus relate it to?
    abbreviations("sailor")                         which letters can this phrase stand for?
    indicator_types("cook")                         which mechanism does this word signal?

`exclude_urls` removes evidence mined from specific puzzles, so an evaluation
never lets the solver look up the very clue it is being tested on.
"""

import re
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from cryptic_agent import config
from cryptic_agent.models import normalize_answer, normalize_phrase


class LexiconNotFoundError(RuntimeError):
    """The lexicon database has not been built yet."""


# The cryptics dataset's mechanism names -> ours (models.ComponentType).
WORDPLAY_NAMES = {
    "anagram": "anagram",
    "container": "container",
    "insertion": "container",  # A in B and B around A are the same mechanism
    "deletion": "deletion",
    "hidden": "hidden_word",
    "homophone": "homophone",
    "reversal": "reversal",
    "alternation": "alternation",  # odd/even letters
}


@dataclass(frozen=True)
class Evidence:
    """A candidate (answer, letters or mechanism) and how many past clues support it."""

    value: str
    support: int
    curated: bool = False


@cache
def curated_indicators() -> dict[str, tuple[str, ...]]:
    """Hand-checked indicators for mechanisms the mined data lacks (e.g. acrostics)."""
    text = (
        resources.files("cryptic_agent.lexicon")
        .joinpath("data/indicators.tsv")
        .read_text(encoding="utf-8")
    )
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            phrase, mechanism = line.split("\t")
            found.setdefault(normalize_phrase(phrase), []).append(mechanism.strip())
    return {phrase: tuple(mechanisms) for phrase, mechanisms in found.items()}


def url_variants(url: str) -> set[str]:
    """Every equivalent spelling of a URL, so excluding a puzzle can't silently miss.

    The cryptics dataset stores 'https://www.fifteensquared.net/...' while the API
    we scrape returns 'https://fifteensquared.net/...': without this, an evaluation
    of an older Guardian clue would not hide the clue's own explanation.
    """
    core = re.sub(r"^https?://(www\.)?", "", url.strip()).rstrip("/")
    return {
        f"{scheme}://{www}{core}{slash}"
        for scheme in ("http", "https")
        for www in ("", "www.")
        for slash in ("", "/")
    }


def default_path() -> Path:
    return config.lexicon_dir() / "lexicon.sqlite"


class Lexicon:
    def __init__(self, path: Path | None = None) -> None:
        path = path or default_path()
        if not path.exists():
            raise LexiconNotFoundError(
                f"lexicon not found at {path}. Run: uv run cryptic-agent ingest"
            )
        self._con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def close(self) -> None:
        self._con.close()

    def _excluding(self, exclude_urls: Collection[str]) -> tuple[str, list[str]]:
        if not exclude_urls:
            return "", []
        urls = sorted({variant for url in exclude_urls for variant in url_variants(url)})
        marks = ",".join("?" * len(urls))
        return (
            f" AND clue_ref NOT IN (SELECT id FROM clue_refs WHERE source_url IN ({marks}))",
            urls,
        )

    def definition_answers(
        self, phrase: str, *, length: int | None = None, exclude_urls: Collection[str] = ()
    ) -> list[Evidence]:
        """Answers this definition led to in past clues, most frequent first."""
        sql = "SELECT answer, COUNT(*) FROM definitions WHERE phrase = ?"
        params: list[object] = [normalize_phrase(phrase)]
        if length is not None:
            sql += " AND length = ?"
            params.append(length)
        extra_sql, extra = self._excluding(exclude_urls)
        rows = self._con.execute(
            sql + extra_sql + " GROUP BY answer ORDER BY COUNT(*) DESC, answer", params + extra
        )
        return [Evidence(answer, n) for answer, n in rows]

    def synonyms(self, phrase: str, *, length: int | None = None) -> list[str]:
        """Thesaurus terms related to `phrase`, as answer letters, alphabetical."""
        sql = "SELECT term FROM thesaurus WHERE phrase = ?"
        params: list[object] = [normalize_phrase(phrase)]
        if length is not None:
            sql += " AND length = ?"
            params.append(length)
        return [term for (term,) in self._con.execute(sql + " ORDER BY term", params)]

    def abbreviations(
        self, phrase: str, *, min_support: int = 3, exclude_urls: Collection[str] = ()
    ) -> list[Evidence]:
        """Letters `phrase` can stand for, e.g. sailor -> AB, TAR.

        Curated entries always count; mined ones only if seen in `min_support`+ clues,
        which filters the dataset's extraction noise (e.g. "a" -> GR).
        """
        extra_sql, extra = self._excluding(exclude_urls)
        rows = self._con.execute(
            "SELECT letters, SUM(curated = 0), MAX(curated) FROM abbreviations "
            "WHERE phrase = ? AND (curated = 1 OR clue_ref IS NOT NULL" + extra_sql + ") "
            "GROUP BY letters HAVING MAX(curated) = 1 OR SUM(curated = 0) >= ? "
            "ORDER BY MAX(curated) DESC, SUM(curated = 0) DESC, letters",
            [normalize_phrase(phrase), *extra, min_support],
        )
        return [Evidence(letters, n, bool(curated)) for letters, n, curated in rows]

    def indicator_types(self, phrase: str, *, exclude_urls: Collection[str] = ()) -> list[Evidence]:
        """Mechanisms this phrase has signalled, most frequent first, in our type names."""
        extra_sql, extra = self._excluding(exclude_urls)
        counts: dict[str, int] = {}
        for wordplay, n in self._con.execute(
            "SELECT wordplay, COUNT(*) FROM indicators WHERE phrase = ?"
            + extra_sql
            + " GROUP BY wordplay",
            [normalize_phrase(phrase), *extra],
        ):
            name = WORDPLAY_NAMES.get(wordplay, wordplay)
            counts[name] = counts.get(name, 0) + n
        mined = [Evidence(t, n) for t, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        curated = [
            Evidence(mechanism, 0, curated=True)
            for mechanism in curated_indicators().get(normalize_phrase(phrase), ())
            if mechanism not in counts
        ]
        return mined + curated

    def extra_words(self) -> list[str]:
        """Words to add to the dictionary: well-attested past answers and repaired UKACD entries."""
        return [word for (word,) in self._con.execute("SELECT word FROM words")]

    def is_known_answer(self, answer: str) -> bool:
        row = self._con.execute(
            "SELECT 1 FROM words WHERE word = ?", (normalize_answer(answer),)
        ).fetchone()
        return row is not None
