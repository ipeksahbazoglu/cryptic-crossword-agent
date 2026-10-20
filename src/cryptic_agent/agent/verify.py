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

A step can work on an earlier step's letters ("documentaries" minus AIM, then an
anagram of DOCUENTRES): its fodder is that step's `produces`. It is checked only as
far as the step it builds on: unchecked letters in, unchecked letters out.
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


def _check_deletion(step: WorksheetStep, feeds_anagram: bool) -> tuple[bool | None, str]:
    """A word with letters taken out: its first or last ('headless'), or another clue
    word's ('documentaries' minus 'aim'). Fodder is the word, then any word removed.

    Letters that a later anagram will shuffle need only be present; otherwise the
    removed letters must be one unbroken run. Never False: the letters may come from
    a synonym of the fodder ('Bird almost' = TER(N)), which code can't see.
    """
    words = [w for w in (normalize_answer(w) for w in step.fodder.split()) if w]
    produces = normalize_answer(step.produces)

    def taken_from(word: str) -> bool:
        return len(word) > len(produces) and not Counter(produces) - Counter(word)

    if len(words) > 2:
        return None, "can't be checked: give the fodder as the word and the word removed"
    if not produces or not any(taken_from(w) for w in words):
        return None, "letters not taken from the fodder as written: perhaps from a synonym"
    if len(words) == 1:
        if produces in {words[0][1:], words[0][:-1], words[0][1:-1]}:
            return True, "first or last letters removed"
        return None, "letters removed some other way: can't be checked by code yet"
    for word, removed in (words, words[::-1]):
        shuffled = feeds_anagram and Counter(word) - Counter(removed) == Counter(produces)
        in_order = removed in word and word.replace(removed, "", 1) == produces
        if (shuffled and not Counter(removed) - Counter(word)) or in_order:
            return True, f"{removed} taken out of {word}"
    return None, "not one word taken out of the other: perhaps an abbreviation"


def _check_step(
    step: WorksheetStep,
    clue: str,
    *,
    builds_on: Check | None = None,
    feeds_anagram: bool = False,
) -> Check:
    """Verify one wordplay step mechanically, where its mechanism allows that.

    `builds_on` is the check of the earlier step whose letters this one works on.
    """
    name = f"wordplay: {step.mechanism} of {step.fodder!r}"
    check = _check_mechanics(
        step, clue, name, chained=builds_on is not None, feeds_anagram=feeds_anagram
    )
    if builds_on is not None and check.passed and not builds_on.passed:
        return Check(
            name=name, passed=None, detail=f"{check.detail}, but builds on an unchecked step"
        )
    return check


def _check_mechanics(
    step: WorksheetStep, clue: str, name: str, *, chained: bool, feeds_anagram: bool
) -> Check:
    literal = step.mechanism in ("anagram", "hidden_word", "acrostic", "alternation")
    if literal and not chained and not _in_clue(clue, step.fodder):
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
    if step.mechanism == "deletion":
        passed, detail = _check_deletion(step, feeds_anagram)
        return Check(name=name, passed=passed, detail=detail)
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
    # Which earlier step's letters each step works on, if any.
    letters = [normalize_answer(s.produces) for s in steps]
    builds_on: list[int | None] = []
    for s in steps:
        fodder = normalize_answer(s.fodder)
        earlier = [j for j in range(len(builds_on)) if letters[j] == fodder and j not in builds_on]
        builds_on.append(earlier[0] if earlier else None)
    feeds_anagram = {
        j
        for s, j in zip(steps, builds_on, strict=True)
        if j is not None and s.mechanism == "anagram"
    }
    step_checks: list[Check] = []
    for i, (s, j) in enumerate(zip(steps, builds_on, strict=True)):
        step_checks.append(
            _check_step(
                s,
                clue,
                builds_on=step_checks[j] if j is not None else None,
                feeds_anagram=i in feeds_anagram,
            )
        )
    checks.extend(step_checks)
    # The answer is built from the steps nothing later works on.
    final = [s for i, s in enumerate(steps) if i not in builds_on]
    final_checks = [c for i, c in enumerate(step_checks) if i not in builds_on]
    if len(final) == 1 and final_checks[0].passed is not False:
        built = normalize_answer(final[0].produces)
        checks.append(
            Check(
                name="wordplay builds the answer",
                passed=built == answer,
                detail=f"{built} vs {answer}",
            )
        )
    elif len(final) > 1:
        joined = "".join(normalize_answer(s.produces) for s in final)
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
    chained = [s for s, j in zip(steps, builds_on, strict=True) if j is not None]
    for s in worksheet.wordplay:
        # A step working on an earlier step's letters takes no clue words as fodder.
        roles += ([] if any(s is c for c in chained) else [s.fodder]) + (
            [s.indicator] if s.indicator else []
        )
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
