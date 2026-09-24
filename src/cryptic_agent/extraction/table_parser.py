"""Read clues out of a blog post's clue table with plain code, no LLM.

Fifteensquared posts lay clues out in a table, and bloggers mark the parts of a
clue with formatting that the scraper keeps (see scraper/clean.py). This module
extracts everything that can be read without judgement:

    | 1 | <color=red>Cook</color> using mere <u>**sweet desserts**</u> (9) | Answer MERINGUES |
    |   | Parsing *anagram* of (USING MERE)\\* with an *anagrind* of "cook" |  |

    -> clue "Cook using mere sweet desserts", enumeration 9, answer MERINGUES,
       definition "sweet desserts", indicator "Cook", type hint "anagram",
       parsing text for the LLM to turn into wordplay components.

Table layouts vary by era and blogger, so cells are recognised by content (the
clue cell contains an enumeration like "(7)"; the answer cell is capitals), not
by column position.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cryptic_agent.models import RawPost, enumeration_lengths, normalize_answer, normalize_phrase

# An enumeration at the end of the clue: "(7)", "(3,4)", "(5-3)", "(2, 3)", "(4)."
_ENUMERATION = re.compile(r"\((\d+(?:\s*[,\-\u2013]\s*\d+)*)\)\s*\.?")
_ANSWER_CELL = re.compile(
    r"^(?:Answer\s+)?([A-Z][A-Za-z' \-]*[A-Z])$"
)  # PiER: case marks changed letters
_UNDERLINE = re.compile(r"<u>(.*?)</u>", re.S)
_COLOR = re.compile(r"<color=[^>]+>(.*?)</color>", re.S)
_ITALIC = re.compile(r"(?<![*\\])\*([^*\n]+?)\*(?!\*)")
_SETTER = re.compile(r"(?:\bby|\u2013|-)\s+([A-Z][\w']+)\s*$")

# How bloggers name mechanisms in italics -> our vocabulary (ComponentType or clue shape).
TYPE_HINT_WORDS: dict[str, str] = {
    "anagram": "anagram",
    "hidden": "hidden_word",
    "hidden word": "hidden_word",
    "reversal": "reversal",
    "reversed": "reversal",
    "reverse": "reversal",
    "soundalike": "homophone",
    "homophone": "homophone",
    "container": "container",
    "insertion": "container",
    "deletion": "deletion",
    "naked word": "deletion",
    "headless": "deletion",
    "curtailment": "deletion",
    "acrostic": "acrostic",
    "initial letters": "acrostic",
    "first letters": "acrostic",
    "charade": "charade",
    "double definition": "double_definition",
    "cryptic definition": "cryptic_definition",
    "&lit": "and_lit",
    "and lit": "and_lit",
}


class ParsedClue(BaseModel):
    """What the table says about one clue, before any LLM interpretation."""

    model_config = ConfigDict(frozen=True)

    number: int
    direction: Literal["across", "down"] | None
    clue_markup: str  # the clue with the blogger's formatting, as scraped
    clue_text: str  # plain clue as printed, without the enumeration
    enumeration: str
    answer: str
    definitions: list[str]  # underlined spans
    indicators: list[str]  # coloured spans
    type_hints: list[str]  # mapped from italic words in the parsing, e.g. ["anagram"]
    parsing: str  # the blogger's explanation
    source_url: str
    source_title: str
    setter: str | None


def strip_markup(text: str) -> str:
    """Plain text: drop <u>, <color>, **bold**, *italic* and markdown escapes."""
    text = re.sub(r"</?u>|<color=[^>]+>|</color>", "", text)
    text = text.replace("\\*", "\x00").replace("*", "").replace("\x00", "*")
    text = re.sub(r"\\([_\[\]#`])", r"\1", text)
    return " ".join(text.replace("\xa0", " ").split())


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _spans(pattern: re.Pattern[str], markup: str) -> list[str]:
    spans = (strip_markup(m) for m in pattern.findall(markup))
    return [s.strip(" ,.;:!?") for s in spans if s.strip(" ,.;:!?")]


def _merge_definition_spans(clue_text: str, spans: list[str]) -> list[str]:
    """Merge definitions that a blogger underlined word by word.

    e.g. "<u>Toddlers'</u> <u>facilities</u>" is one definition, not two.

    A definition sits at the start or end of the clue, so a span that is at
    neither is merged with its neighbours; spans that already sit at an end are
    kept apart (a double definition underlines one at each end).
    """
    clue = normalize_phrase(clue_text)

    def at_an_end(span: str) -> bool:
        s = normalize_phrase(span)
        return clue.startswith(s) or clue.endswith(s)

    if all(at_an_end(s) for s in spans):
        return spans
    merged = " ".join(spans)
    return [merged] if at_an_end(merged) else spans


def type_hints_from(parsing: str) -> list[str]:
    hints: list[str] = []
    for phrase in _ITALIC.findall(parsing):
        words = normalize_phrase(phrase).removesuffix(" of")
        mapped = TYPE_HINT_WORDS.get(words) or TYPE_HINT_WORDS.get(phrase.strip().lower())
        if mapped and mapped not in hints:
            hints.append(mapped)
    return hints


def _setter_from(title: str) -> str | None:
    match = _SETTER.search(title)
    return match.group(1) if match else None


def _parse_clue_cell(cell: str) -> tuple[str, str, str] | None:
    """Split a clue cell into (clue markup, enumeration, trailing text) or None."""
    matches = list(_ENUMERATION.finditer(cell))
    if not matches:
        return None
    enum = matches[0]  # the first enumeration ends the clue; later ones are in the parsing
    clue_markup = cell[: enum.start()].strip()
    enumeration = re.sub(r"\s", "", enum.group(1)).replace("\u2013", "-")
    return clue_markup, enumeration, cell[enum.end() :].strip()


def parse_clue_table(post: RawPost) -> list[ParsedClue]:
    """All clues found in `post`'s tables that pass basic consistency checks."""
    clues: list[ParsedClue] = []
    direction: Literal["across", "down"] | None = None
    lines = [line for line in post.content_markdown.splitlines() if line.startswith("|")]

    for i, line in enumerate(lines):
        cells = _cells(line)
        header = strip_markup(" ".join(cells)).upper()
        if re.search(r"\bACROSS\b", header) and not re.search(r"\(\d", header):
            direction = "across"
            continue
        if re.search(r"\bDOWN\b", header) and not re.search(r"\(\d", header):
            direction = "down"
            continue

        number_match = re.match(r"^(\d+)", cells[0]) if cells else None
        if number_match is None:
            continue

        clue_part = answer = parsing = None
        for cell in cells[1:]:
            if clue_part is None and (parsed := _parse_clue_cell(cell)):
                clue_part = parsed
            elif answer is None and (m := _ANSWER_CELL.match(strip_markup(cell))):
                answer = m.group(1)
        if clue_part is None or answer is None:
            continue
        clue_markup, enumeration, trailing = clue_part

        if trailing:  # early layout: the parsing follows the enumeration in the same cell
            parsing = trailing
        elif i + 1 < len(lines) and (nxt := _cells(lines[i + 1])) and not nxt[0]:
            parsing = next((c for c in nxt[1:] if c), "")
        parsing = re.sub(r"^Parsing\s+", "", parsing or "")

        clue_text = strip_markup(clue_markup)
        if len(normalize_answer(answer)) != sum(enumeration_lengths(enumeration)):
            continue  # misread row (or a typo in the post): skip rather than guess

        clues.append(
            ParsedClue(
                number=int(number_match.group(1)),
                direction=direction,
                clue_markup=clue_markup,
                clue_text=clue_text,
                enumeration=enumeration,
                answer=normalize_answer(answer),
                definitions=_merge_definition_spans(clue_text, _spans(_UNDERLINE, clue_markup)),
                indicators=_spans(_COLOR, clue_markup),
                type_hints=type_hints_from(parsing),
                parsing=strip_markup(parsing) if parsing else "",
                source_url=post.url,
                source_title=post.title,
                setter=_setter_from(post.title),
            )
        )
    return clues
