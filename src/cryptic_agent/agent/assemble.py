"""Tier 0: solve a clue with code alone, when the fast pass already explains it.

If the fast pass found a STRONG candidate (a definition at one end, wordplay from
separate words), code can often write the whole worksheet itself. That costs no
tokens, so it is tried first, under rules that keep it from being over-confident:

  - the wordplay must have a recognised indicator next to it,
  - every remaining clue word must be a genuine link word ("a", "for", "in", ...),
  - the worksheet must pass the same verify() as any model-written one,
  - and exactly one answer may survive: if two different answers can both be
    assembled and verified, code does not guess; the clue goes to the model.
"""

from collections.abc import Collection
from typing import Literal

from cryptic_agent.agent.fastpass import Candidate, FastPass, Span, clue_words
from cryptic_agent.agent.verify import Verdict, verify
from cryptic_agent.agent.worksheet import Worksheet, WorksheetDefinition, WorksheetStep
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.models import normalize_phrase
from cryptic_agent.tools.dictionary import Dictionary

# Words that only connect definition and wordplay. Deliberately short: a word that
# could plausibly be an indicator or part of a definition does not belong here.
LINK_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "being", "by", "for", "from",
        "gets", "getting", "gives", "giving", "has", "have", "in", "is", "it", "its",
        "makes", "making", "of", "on", "or", "provides", "s", "shows", "so", "that",
        "the", "this", "to", "was", "when", "where", "which", "while", "with",
    }
)  # fmt: skip

EXPLANATIONS = {
    "anagram": "anagram of {fodder}",
    "hidden_word": "hidden in {fodder}",
    "reversal": "{fodder} reversed",
    "acrostic": "first or last letters of {fodder}",
    "alternation": "alternate letters of {fodder}",
}


def _covered(span: Span | None) -> set[int]:
    return set(range(span.start, span.end)) if span else set()


def _worksheets(candidate: Candidate, words: list[str]) -> list[Worksheet]:
    """Every complete worksheet code can write for this candidate."""
    sheets: list[Worksheet] = []
    for definition in candidate.definitions:
        for wordplay in candidate.wordplay:
            if wordplay.indicator is None or definition.span.overlaps(wordplay.span):
                continue
            if wordplay.indicator_span and definition.span.overlaps(wordplay.indicator_span):
                continue
            used = (
                _covered(definition.span)
                | _covered(wordplay.span)
                | _covered(wordplay.indicator_span)
            )
            leftover = [words[i] for i in range(len(words)) if i not in used]
            leftover_words = [w for w in (normalize_phrase(x) for x in leftover) if w]
            if any(part not in LINK_WORDS for w in leftover_words for part in w.split()):
                continue
            definition_text = " ".join(words[definition.span.start : definition.span.end])
            position: Literal["start", "end"] = "start" if definition.span.start == 0 else "end"
            sheets.append(
                Worksheet(
                    answer=candidate.answer,
                    definitions=[WorksheetDefinition(text=definition_text, position=position)],
                    wordplay=[
                        WorksheetStep(
                            mechanism=wordplay.mechanism,  # type: ignore[arg-type]
                            indicator=wordplay.indicator,
                            fodder=wordplay.fodder,
                            produces=candidate.answer,
                            explanation=EXPLANATIONS[wordplay.mechanism].format(
                                fodder=repr(wordplay.fodder)
                            )
                            + f", indicated by {wordplay.indicator!r}",
                        )
                    ],
                    link_words=leftover,
                    confidence="confirmed",
                    alternatives=[],
                )
            )
    return sheets


def solve_by_code(
    found: FastPass,
    dictionary: Dictionary,
    lexicon: Lexicon,
    *,
    pattern: str | None = None,
    exclude_urls: Collection[str] = (),
) -> tuple[Worksheet, Verdict] | None:
    """A confirmed worksheet if code alone explains the clue unambiguously, else None."""
    words = clue_words(found.clue)
    confirmed: dict[str, tuple[Worksheet, Verdict]] = {}
    for candidate in found.strong:
        for sheet in _worksheets(candidate, words):
            verdict = verify(
                sheet,
                found.clue,
                found.enumeration,
                dictionary,
                pattern=pattern,
                lexicon=lexicon,
                exclude_urls=exclude_urls,
            )
            if verdict.status == "confirmed":
                confirmed.setdefault(verdict.answer, (sheet, verdict))
    if len(confirmed) == 1:
        return next(iter(confirmed.values()))
    return None  # nothing assembled, or ambiguous: let the model decide
