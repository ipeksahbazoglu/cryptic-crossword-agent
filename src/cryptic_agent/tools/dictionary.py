"""A word list with fast membership and anagram lookups.

The solver's tools take a Dictionary as an argument instead of loading a global
word list at import time, so tests can use a small in-memory one and the real
source (UKACD, see lexicon/wordlist.py) is loaded in one place.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from cryptic_agent import config
from cryptic_agent.lexicon.sources import UKACD
from cryptic_agent.lexicon.wordlist import read_ukacd
from cryptic_agent.models import normalize_answer

logger = logging.getLogger(__name__)


class DictionaryNotFoundError(RuntimeError):
    """The word list's data files are not installed."""


def _letter_key(word: str) -> str:
    """Anagrams share a key: 'SENATOR' and 'TREASON' both map to 'AENORST'."""
    return "".join(sorted(word))


class Dictionary:
    def __init__(self, words: Iterable[str]) -> None:
        normalized = {normalize_answer(w) for w in words}
        normalized.discard("")
        self._words = frozenset(normalized)
        by_key: defaultdict[str, list[str]] = defaultdict(list)
        for word in sorted(self._words):
            by_key[_letter_key(word)].append(word)
        self._by_letter_key = dict(by_key)

    @classmethod
    def from_ukacd(cls, zip_path: Path | None = None) -> "Dictionary":
        """Load UKACD: every entry, plus each word of its phrases.

        'no sweat' is stored as NOSWEAT (a whole answer) and as NO and SWEAT, so
        check_answer can verify a (2,5) answer word by word.
        """
        path = zip_path or config.lexicon_dir() / "sources" / UKACD.filename
        if not path.exists():
            raise DictionaryNotFoundError(
                f"UKACD word list not found at {path}. Run: uv run cryptic-agent ingest"
            )
        ukacd = read_ukacd(path)
        logger.info(
            "UKACD: %d entries; %d with lost accents set aside",
            len(ukacd.entries),
            len(ukacd.damaged),
        )
        phrase_words = (word for entry in ukacd.entries if " " in entry for word in entry.split())
        return cls([*ukacd.entries, *phrase_words])

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and normalize_answer(word) in self._words

    def __len__(self) -> int:
        return len(self._words)

    def anagrams(self, letters: str) -> list[str]:
        """All dictionary words using exactly these letters, in alphabetical order."""
        return list(self._by_letter_key.get(_letter_key(normalize_answer(letters)), []))
