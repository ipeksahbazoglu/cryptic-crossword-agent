"""A word list with fast membership and anagram lookups.

The solver's tools take a Dictionary as an argument instead of loading a global
word list at import time, so tests can use a small in-memory one and the real
source (NLTK today, maybe a Collins list later) can be swapped in one place.
"""

from collections import defaultdict
from collections.abc import Iterable

from cryptic_agent.models import normalize_answer


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
    def from_nltk(cls) -> "Dictionary":
        """Load NLTK's English `words` corpus (about 234k words)."""
        from nltk.corpus import words  # imported here: nltk is slow to import

        try:
            return cls(words.words())
        except LookupError as exc:
            raise DictionaryNotFoundError(
                "NLTK 'words' corpus not found. Run: "
                "uv run python -c \"import nltk; nltk.download('words')\""
            ) from exc

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and normalize_answer(word) in self._words

    def __len__(self) -> int:
        return len(self._words)

    def anagrams(self, letters: str) -> list[str]:
        """All dictionary words using exactly these letters, in alphabetical order."""
        return list(self._by_letter_key.get(_letter_key(normalize_answer(letters)), []))
