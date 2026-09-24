"""Mechanical checks the solver uses to verify a hypothesis instead of trusting it.

An LLM saying "TREASON is an anagram of SENATOR" is far more trustworthy once
something has checked the letters really match and TREASON is a real word.
Each function is pure: same inputs, same output, no network or globals.
"""

import re

from pydantic import BaseModel, ConfigDict

from cryptic_agent.models import enumeration_lengths, normalize_answer
from cryptic_agent.tools.dictionary import Dictionary

MAX_ANAGRAM_MATCHES = 50


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)


class AnagramResult(ToolResult):
    input_letters: str
    matches: list[str]
    count: int


class HiddenWord(ToolResult):
    word: str
    spans_words: bool  # True for a classic hidden word split across clue words


class HiddenWordResult(ToolResult):
    squashed_text: str
    matches: list[HiddenWord]


class AnswerCheck(ToolResult):
    answer: str
    enumeration: str
    length_ok: bool
    unknown_words: list[str]  # parts of the answer not found in the dictionary
    valid: bool


class ReversalResult(ToolResult):
    input: str
    reversed: str


def find_anagrams(dictionary: Dictionary, letters: str, length: int | None = None) -> AnagramResult:
    """Dictionary words that are anagrams of `letters`, excluding `letters` itself.

    In a clue the answer is never the anagram fodder unchanged, so the input
    word is left out of the matches.
    """
    target = normalize_answer(letters)
    if length is not None and length != len(target):
        return AnagramResult(input_letters=target, matches=[], count=0)
    matches = [w for w in dictionary.anagrams(target) if w != target]
    return AnagramResult(
        input_letters=target, matches=matches[:MAX_ANAGRAM_MATCHES], count=len(matches)
    )


def check_hidden_word(dictionary: Dictionary, text: str, length: int) -> HiddenWordResult:
    """Dictionary words of `length` letters hidden in `text`, ignoring spaces.

    'a Roman cerebral' hides ROMANCE across 'Roman' and 'cerebral'. Matches that
    span two or more clue words come first. A match that is a whole clue word on
    its own is not hidden at all, so it is excluded.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    clue_words = [normalize_answer(w) for w in re.split(r"\s+", text)]
    clue_words = [w for w in clue_words if w]
    squashed = "".join(clue_words)

    # word_index[i] = which clue word the i-th letter of `squashed` came from
    word_index = [i for i, word in enumerate(clue_words) for _ in word]
    whole_words = set(clue_words)

    found: dict[str, bool] = {}
    for start in range(len(squashed) - length + 1):
        candidate = squashed[start : start + length]
        if candidate not in dictionary:
            continue
        spans = word_index[start] != word_index[start + length - 1]
        if not spans and candidate in whole_words:
            continue
        found[candidate] = found.get(candidate, False) or spans

    matches = [
        HiddenWord(word=w, spans_words=s)
        for w, s in sorted(found.items(), key=lambda item: (not item[1], item[0]))
    ]
    return HiddenWordResult(squashed_text=squashed, matches=matches)


def check_answer(dictionary: Dictionary, answer: str, enumeration: str) -> AnswerCheck:
    """Check a candidate fits the enumeration and every word in it is real.

    Multi-word answers are split by the enumeration, so ICECREAM (3,5) is
    checked as ICE + CREAM rather than looked up as one word.
    """
    letters = normalize_answer(answer)
    lengths = enumeration_lengths(enumeration)
    length_ok = len(letters) == sum(lengths)

    unknown: list[str] = []
    if length_ok:
        start = 0
        for n in lengths:
            part = letters[start : start + n]
            if part not in dictionary:
                unknown.append(part)
            start += n
    elif letters not in dictionary:
        unknown.append(letters)

    return AnswerCheck(
        answer=letters,
        enumeration=enumeration,
        length_ok=length_ok,
        unknown_words=unknown,
        valid=length_ok and not unknown,
    )


def reverse_letters(text: str) -> ReversalResult:
    """Reverse the letters of `text`, for reversal clues: 'STRAP' -> 'PARTS'."""
    letters = normalize_answer(text)
    return ReversalResult(input=letters, reversed=letters[::-1])
