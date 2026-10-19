"""Check the model's worksheet with code, and decide the verdict.

The model's own confidence is never enough. An answer is CONFIRMED only when:
  - it fits the enumeration (and crossing-letter pattern) and is a real word,
  - every definition sits where the worksheet says (start, end or whole clue),
  - at least one wordplay step is verified mechanically and none fails,
  - every clue word has exactly one role (definition, indicator, fodder or link),
  - if the wordplay could also have made a different real word (SORE, ORES and
    ROES are all anagrams of "rose"), the lexicon links the definition to this one,
  - and the model itself is not unsure.
Otherwise it is PENCILLED (fits, but not proven) or UNSURE (doesn't even fit).

A check's `passed` is True, False, or None for "code can't tell" (e.g. a charade
built from synonyms): None never confirms, but never blocks either.
"""

from collections import Counter
from collections.abc import Collection
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cryptic_agent.agent.worksheet import Worksheet, WorksheetStep
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.models import normalize_answer, normalize_phrase
from cryptic_agent.tools.dictionary import Dictionary
from cryptic_agent.tools.wordplay import LetterSelection, check_answer, select_letters

Status = Literal["confirmed", "pencilled", "unsure"]


class Check(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool | None
    detail: str


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    status: Status
    checks: list[Check]

    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]


def matches_pattern(answer: str, pattern: str) -> bool:
    """'M?R?N?U?S' (or with _ or .) against an answer; pattern letters must match."""
    letters = normalize_answer(answer)
    slots = [c for c in pattern.upper() if c.isalpha() or c in "?_."]
    return len(slots) == len(letters) and all(
        s in "?_." or s == a for s, a in zip(slots, letters, strict=True)
    )


def _in_clue(clue: str, phrase: str) -> bool:
    """Whole clue words. 'of ... for' (a split indicator) means each piece is in the clue."""
    pieces = [p for p in phrase.replace("\u2026", "...").split("...") if p.strip()]
    return bool(pieces) and all(
        f" {normalize_phrase(p)} " in f" {normalize_phrase(clue)} " for p in pieces
    )


def _check_step(step: WorksheetStep, clue: str) -> Check:
    """Verify one wordplay step mechanically, where its mechanism allows that."""
    name = f"wordplay: {step.mechanism} of {step.fodder!r}"
    literal = step.mechanism in ("anagram", "hidden_word", "acrostic", "alternation")
    if literal and not _in_clue(clue, step.fodder):
        # These mechanisms use clue words verbatim; others may use a synonym.
        return Check(name=name, passed=False, detail="fodder is not words of the clue")
    if step.indicator and not _in_clue(clue, step.indicator):
        return Check(name=name, passed=False, detail=f"indicator {step.indicator!r} not in clue")

    fodder, produces = normalize_answer(step.fodder), normalize_answer(step.produces)
    if step.mechanism == "anagram":
        ok = sorted(fodder) == sorted(produces) and fodder != produces
        return Check(name=name, passed=ok, detail="letters rearrange" if ok else "letters differ")
    if step.mechanism == "hidden_word":
        ok = produces in fodder or produces[::-1] in fodder
        return Check(name=name, passed=ok, detail="hidden there" if ok else "not hidden there")
    if step.mechanism in ("acrostic", "alternation"):
        ways: tuple[LetterSelection, ...] = (
            ("first", "last") if step.mechanism == "acrostic" else ("odd", "even")
        )
        ok = produces in {select_letters(step.fodder, way).letters for way in ways}
        return Check(name=name, passed=ok, detail="letters match" if ok else "letters differ")
    if step.mechanism == "reversal" and fodder[::-1] == produces:
        return Check(name=name, passed=True, detail="fodder reversed")
    return Check(name=name, passed=None, detail="can't be checked by code yet")


def _other_words(step: WorksheetStep, answer: str, dictionary: Dictionary) -> list[str]:
    """Other real words the same wordplay could have produced from the same fodder."""
    fodder, n = normalize_answer(step.fodder), len(answer)
    if step.mechanism == "anagram":
        candidates = set(dictionary.anagrams(fodder)) - {fodder}
    elif step.mechanism == "hidden_word":
        pieces = {fodder[i : i + n] for i in range(len(fodder) - n + 1)}
        candidates = {w for piece in pieces for w in (piece, piece[::-1]) if w in dictionary}
    elif step.mechanism in ("acrostic", "alternation"):
        ways: tuple[LetterSelection, ...] = (
            ("first", "last") if step.mechanism == "acrostic" else ("odd", "even")
        )
        picked = {select_letters(step.fodder, way).letters for way in ways}
        candidates = {w for w in picked if len(w) == n and w in dictionary}
    else:
        return []
    return sorted(candidates - {answer})


def _definition_supports(
    lexicon: Lexicon, text: str, answer: str, exclude_urls: Collection[str]
) -> bool:
    """Does the lexicon (past clues or thesaurus) link this definition to the answer?"""
    text = text.strip(' ,.;:!?"')
    variants = {text, text.removesuffix("'s").removesuffix("\u2019s")}  # 's is often "is"
    for variant in variants:
        past = lexicon.definition_answers(variant, length=len(answer), exclude_urls=exclude_urls)
        if answer in {e.value for e in past} or answer in lexicon.synonyms(
            variant, length=len(answer)
        ):
            return True
    return False


def verify(
    worksheet: Worksheet,
    clue: str,
    enumeration: str,
    dictionary: Dictionary,
    *,
    pattern: str | None = None,
    lexicon: Lexicon | None = None,
    exclude_urls: Collection[str] = (),
) -> Verdict:
    answer = normalize_answer(worksheet.answer)
    checks: list[Check] = []

    fit = check_answer(dictionary, answer, enumeration)
    checks.append(
        Check(
            name="answer fits",
            passed=fit.length_ok,
            detail=f"{len(answer)} letters for ({enumeration})",
        )
    )
    checks.append(
        Check(
            name="real word",
            passed=fit.valid if fit.length_ok else None,
            detail="in the dictionary" if fit.valid else f"unknown: {fit.unknown_words}",
        )
    )
    if pattern:
        checks.append(
            Check(name="crossing letters", passed=matches_pattern(answer, pattern), detail=pattern)
        )

    clue_phrase = normalize_phrase(clue)
    for d in worksheet.definitions:
        text = normalize_phrase(d.text)
        at = {
            "start": clue_phrase.startswith(text + " ") or clue_phrase == text,
            "end": clue_phrase.endswith(" " + text) or clue_phrase == text,
            "whole": clue_phrase == text,
        }[d.position]
        checks.append(
            Check(
                name=f"definition {d.text!r}",
                passed=bool(text) and at,
                # Worded to make sense on its own: tier 2 shows failures to the model.
                detail=(
                    f"at the {d.position}"
                    if bool(text) and at
                    else f"not at the {d.position} of the clue"
                ),
            )
        )
    if not worksheet.definitions:
        checks.append(Check(name="definition", passed=False, detail="no definition given"))

    steps = [
        s
        for s in worksheet.wordplay
        if s.mechanism not in ("double_definition", "cryptic_definition")
    ]
    step_checks = [_check_step(s, clue) for s in steps]
    checks.extend(step_checks)
    if len(steps) == 1 and step_checks[0].passed is not False:
        built = normalize_answer(steps[0].produces)
        checks.append(
            Check(
                name="wordplay builds the answer",
                passed=built == answer,
                detail=f"{built} vs {answer}",
            )
        )
    elif len(steps) > 1:
        joined = "".join(normalize_answer(s.produces) for s in steps)
        checks.append(
            Check(
                name="wordplay builds the answer",
                passed=True if joined == answer else None,
                detail="pieces join to the answer"
                if joined == answer
                else "pieces combine non-linearly",
            )
        )

    roles = [d.text for d in worksheet.definitions] + worksheet.link_words
    for s in worksheet.wordplay:
        roles += [s.fodder] + ([s.indicator] if s.indicator else [])
    # Count words, so a repeated clue word needs a role each time it appears, and no
    # word can serve twice (as definition and fodder, say). An &lit clue is the
    # exception: the whole clue is the definition and also the wordplay.
    in_clue = Counter(clue_phrase.split())
    in_roles = Counter(" ".join(normalize_phrase(r) for r in roles).split())
    and_lit = any(d.position == "whole" for d in worksheet.definitions)
    missing = sorted((in_clue - in_roles).elements())
    reused = [] if and_lit else sorted((in_roles - in_clue).elements())
    problems = [f"unexplained: {missing}"] * bool(missing) + [
        f"used more often than it appears: {reused}"
    ] * bool(reused)
    checks.append(
        Check(
            name="every word has a role",
            passed=not problems,
            detail="; ".join(problems) or "all words accounted for",
        )
    )

    # If the same wordplay could have made a different real word, the mechanics
    # alone don't pick this answer: the definition has to.
    verified = [s for s, c in zip(steps, step_checks, strict=True) if c.passed]
    others = sorted({w for s in verified for w in _other_words(s, answer, dictionary)})
    if others:
        supported = lexicon is not None and any(
            _definition_supports(lexicon, d.text, answer, exclude_urls)
            for d in worksheet.definitions
        )
        checks.append(
            Check(
                name="definition means the answer",
                passed=True if supported else None,
                detail="found in past clues or the thesaurus"
                if supported
                else f"not found; the wordplay also gives {', '.join(others[:5])}",
            )
        )

    return Verdict(answer=answer, status=_status(worksheet, checks, step_checks), checks=checks)


def _status(worksheet: Worksheet, checks: list[Check], step_checks: list[Check]) -> Status:
    by_name = {c.name: c for c in checks}
    pattern = by_name.get("crossing letters")
    if not by_name["answer fits"].passed or (pattern is not None and not pattern.passed):
        return "unsure"
    if any(c.passed is False for c in checks) or worksheet.confidence != "confirmed":
        return "pencilled"
    meaning = by_name.get("definition means the answer")
    if meaning is not None and not meaning.passed:
        return "pencilled"  # ambiguous wordplay and nothing ties the definition to it
    # Double and cryptic definitions have no mechanics to check: pencilled until
    # code can also check that each definition really means the answer.
    if by_name["real word"].passed and any(c.passed for c in step_checks):
        return "confirmed"
    return "pencilled"
