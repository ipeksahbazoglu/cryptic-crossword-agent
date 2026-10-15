"""The fast pass: what code can find before the model starts thinking.

Like an expert's first glance at a clue, it tries the quick wins mechanically:

  definition side  spans of 1-4 words at the start or end -> past answers for that
                   definition and thesaurus terms, of the right length
  wordplay side    from the clue's own letters: anagrams of consecutive words with
                   exactly the right letter count, hidden words (both directions),
                   first/last letters, alternate letters, literal reversals; each
                   tagged with any neighbouring indicator the lexicon recognises
  the "aha"        a candidate is STRONG when a definition span and a separate
                   wordplay span produce the same word

The model gets these as evidence in its first message. It still has to fill in the
worksheet, and verify.py still decides the verdict: the fast pass only gives the
model a head start, it never confirms anything by itself.
"""

from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

from cryptic_agent.agent.verify import matches_pattern
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.models import enumeration_lengths, normalize_answer
from cryptic_agent.tools.dictionary import Dictionary
from cryptic_agent.tools.wordplay import LetterSelection, select_letters

MAX_DEFINITION_WORDS = 4
MAX_INDICATOR_WORDS = 3
MAX_CANDIDATES_SHOWN = 8


@dataclass(frozen=True)
class Span:
    """Clue words [start, end) by position."""

    start: int
    end: int

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class DefinitionHit:
    answer: str
    text: str  # the definition words
    span: Span
    source: str  # "past clues (n)" or "thesaurus"


@dataclass(frozen=True)
class WordplayHit:
    answer: str
    mechanism: str
    fodder: str
    span: Span
    indicator: str | None  # a neighbouring phrase the lexicon knows signals this mechanism


@dataclass
class Candidate:
    answer: str
    definitions: list[DefinitionHit] = field(default_factory=list)
    wordplay: list[WordplayHit] = field(default_factory=list)

    @property
    def strong(self) -> bool:
        """A definition and a separate (non-overlapping) wordplay agree."""
        return any(not d.span.overlaps(w.span) for d in self.definitions for w in self.wordplay)

    def score(self) -> tuple[int, int, int, str]:
        has_indicator = any(w.indicator for w in self.wordplay)
        return (-int(self.strong), -int(has_indicator), -len(self.definitions), self.answer)


@dataclass
class FastPass:
    clue: str
    enumeration: str
    candidates: list[Candidate]

    @property
    def strong(self) -> list[Candidate]:
        return [c for c in self.candidates if c.strong]

    def summary(self) -> str:
        """Compact evidence for the model's first message."""
        if not self.candidates:
            return "Fast pass: no quick wins found mechanically; solve from scratch."
        lines = ["Fast pass (found mechanically; verify before relying on it):"]
        for c in self.candidates[:MAX_CANDIDATES_SHOWN]:
            tag = "STRONG" if c.strong else "partial"
            parts = []
            for d in c.definitions[:2]:
                parts.append(f"definition {d.text!r} ({d.source})")
            for w in c.wordplay[:2]:
                via = f", indicator {w.indicator!r}" if w.indicator else ""
                parts.append(f"{w.mechanism} of {w.fodder!r}{via}")
            lines.append(f"- {c.answer} [{tag}]: " + "; ".join(parts))
        return "\n".join(lines)


def _words(clue: str) -> list[str]:
    return [w for w in clue.split() if normalize_answer(w)]


def _phrase(words: list[str], span: Span) -> str:
    return " ".join(words[span.start : span.end])


def _fits(dictionary: Dictionary, answer: str, enumeration: str, pattern: str | None) -> bool:
    """A generated candidate must be a whole dictionary entry of the right length.

    Stricter than check_answer on purpose: splitting arbitrary letters by the
    enumeration finds "words" by accident (LOCUMOP as LOCUM + OP for 5-2), while
    real phrases (STAND-IN, ICE CREAM) are stored whole in UKACD.
    """
    if pattern and not matches_pattern(answer, pattern):
        return False
    return len(answer) == sum(enumeration_lengths(enumeration)) and answer in dictionary


def _definition_hits(
    words: list[str],
    length: int,
    lexicon: Lexicon,
    exclude_urls: Collection[str],
) -> Iterator[DefinitionHit]:
    n = len(words)
    spans = {Span(0, k) for k in range(1, min(MAX_DEFINITION_WORDS, n - 1) + 1)}
    spans |= {Span(n - k, n) for k in range(1, min(MAX_DEFINITION_WORDS, n - 1) + 1)}
    for span in sorted(spans, key=lambda s: (s.start, s.end)):
        raw = _phrase(words, span).strip(",.;:!?'\"")
        # A trailing 's often means "is"/"has" ("Love god's sparkling rose" =
        # "Love god is ..."), so also try the span without it.
        texts = [raw]
        for suffix in ("'s", "’s"):
            if raw.endswith(suffix):
                texts.append(raw.removesuffix(suffix))
        for text in texts:
            for e in lexicon.definition_answers(text, length=length, exclude_urls=exclude_urls):
                yield DefinitionHit(e.value, text, span, f"past clues ({e.support})")
            for term in lexicon.synonyms(text, length=length):
                yield DefinitionHit(term, text, span, "thesaurus")


def _indicator_near(
    words: list[str], span: Span, mechanism: str, lexicon: Lexicon, exclude_urls: Collection[str]
) -> str | None:
    """A phrase right before or after `span` that has signalled `mechanism` before."""
    for size in range(MAX_INDICATOR_WORDS, 0, -1):
        for start in (span.start - size, span.end):
            near = Span(start, start + size)
            if near.start < 0 or near.end > len(words):
                continue
            text = _phrase(words, near).strip(",.;:!?'\"")
            types = lexicon.indicator_types(text, exclude_urls=exclude_urls)
            if any(e.value == mechanism for e in types):
                return text
    return None


def _wordplay_hits(
    words: list[str],
    enumeration: str,
    dictionary: Dictionary,
    lexicon: Lexicon,
    pattern: str | None,
    exclude_urls: Collection[str],
) -> Iterator[WordplayHit]:
    length = sum(enumeration_lengths(enumeration))
    n = len(words)

    def hit(answer: str, mechanism: str, span: Span) -> WordplayHit | None:
        if not _fits(dictionary, answer, enumeration, pattern):
            return None
        indicator = _indicator_near(words, span, mechanism, lexicon, exclude_urls)
        return WordplayHit(answer, mechanism, _phrase(words, span), span, indicator)

    for start in range(n):
        for end in range(start + 1, n + 1):
            span = Span(start, end)
            fodder = _phrase(words, span)
            letters = normalize_answer(fodder)
            # No early exit on run length: acrostics take one letter per word and
            # hidden words can sit in long words, so each mechanism checks its own
            # limits. Clues are short (~4-12 words), so trying every run is cheap.

            # anagram: exactly the right letters
            if len(letters) == length:
                for word in dictionary.anagrams(letters):
                    if word != letters and (h := hit(word, "anagram", span)):
                        yield h
                if (h := hit(letters[::-1], "reversal", span)) and h.answer != letters:
                    yield h

            # hidden word: must cross a word boundary, starting in the run's first
            # word and ending in its last (so each run reports only its own hits)
            if end - start >= 2 and len(letters) > length:
                first_word_ends = len(normalize_answer(words[start]))
                last_word_starts = len(letters) - len(normalize_answer(words[end - 1]))
                for i in range(len(letters) - length + 1):
                    if i >= first_word_ends or i + length <= last_word_starts:
                        continue
                    piece = letters[i : i + length]
                    for candidate in (piece, piece[::-1]):
                        if h := hit(candidate, "hidden_word", span):
                            yield h

            # letter selection: one letter per word, or alternate letters
            if end - start == length:
                ways: tuple[LetterSelection, ...] = ("first", "last")
                for way in ways:
                    picked = select_letters(fodder, way).letters
                    if h := hit(picked, "acrostic", span):
                        yield h
            if len(letters) in (2 * length, 2 * length - 1, 2 * length + 1):
                alternations: tuple[LetterSelection, ...] = ("odd", "even")
                for way in alternations:
                    picked = select_letters(fodder, way).letters
                    if len(picked) == length and (h := hit(picked, "alternation", span)):
                        yield h


def fast_pass(
    clue: str,
    enumeration: str,
    dictionary: Dictionary,
    lexicon: Lexicon,
    *,
    pattern: str | None = None,
    exclude_urls: Collection[str] = (),
) -> FastPass:
    words = _words(clue)
    length = sum(enumeration_lengths(enumeration))
    candidates: dict[str, Candidate] = {}

    for d in _definition_hits(words, length, lexicon, exclude_urls):
        answer = normalize_answer(d.answer)
        if pattern and not matches_pattern(answer, pattern):
            continue
        candidates.setdefault(answer, Candidate(answer)).definitions.append(d)

    seen: set[tuple[str, str, Span]] = set()
    for w in _wordplay_hits(words, enumeration, dictionary, lexicon, pattern, exclude_urls):
        if (w.answer, w.mechanism, w.span) not in seen:
            seen.add((w.answer, w.mechanism, w.span))
            candidates.setdefault(w.answer, Candidate(w.answer)).wordplay.append(w)

    # Keep candidates with wordplay evidence, or definition-only ones from past clues
    # (thesaurus-only lists are long and weak on their own).
    kept = [
        c
        for c in candidates.values()
        if c.wordplay or any(d.source.startswith("past") for d in c.definitions)
    ]
    kept.sort(key=Candidate.score)
    return FastPass(clue, enumeration, kept)
