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

# How bloggers name mechanisms -> our vocabulary (see agent/worksheet.py Mechanism).
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
    """Mechanisms the blogger names: in italics anywhere, or else as the opening words.

    Many bloggers italicise the mechanism ("*anagram* of ..."); others start the
    explanation with it in plain text ("An anagram (diverted) of ...", "Double
    definition"). Plain-text names later in an explanation are ignored, since
    they are often asides ("the anagram indicator fits the surface").
    """
    hints: list[str] = []
    for phrase in _ITALIC.findall(parsing):
        words = normalize_phrase(phrase).removesuffix(" of")
        mapped = TYPE_HINT_WORDS.get(words) or TYPE_HINT_WORDS.get(phrase.strip().lower())
        if mapped and mapped not in hints:
            hints.append(mapped)
    if hints:
        return hints
    opening = normalize_phrase(parsing).split()[:3]
    if opening and opening[0] in ("a", "an"):
        opening = opening[1:]
    for size in (2, 1):
        if mapped := TYPE_HINT_WORDS.get(" ".join(opening[:size])):
            return [mapped]
    return []


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


_WHOLE_COLOR = re.compile(r"^\s*<color=[^>]+>(.*)</color>\s*$", re.S)


def _unwrap_whole_clue_color(markup: str) -> str:
    """Some bloggers colour the whole clue (not just the indicator): drop that wrapper.

    'Quick Cryptic' posts colour only the indicator red; others wrap every clue in
    blue. A colour around the entire clue says nothing about indicators.
    """
    match = _WHOLE_COLOR.match(markup)
    if match and "</color>" not in match.group(1):
        return match.group(1)
    return markup


def _indicator_spans(clue_markup: str, clue_text: str, definitions: list[str]) -> list[str]:
    """Coloured spans that mark indicators, not the blogger colouring something else.

    A colour covering the whole clue, or wrapping an underlined definition, is a
    blogger's styling of the clue or its definition, not an indicator.
    """
    clue = normalize_phrase(clue_text)
    defs = [normalize_phrase(d) for d in definitions]
    return [
        span
        for span in _spans(_COLOR, clue_markup)
        if normalize_phrase(span) != clue
        and not any(d and f" {d} " in f" {normalize_phrase(span)} " for d in defs)
    ]


def _make_clue(
    post: RawPost,
    number: int,
    direction: Literal["across", "down"] | None,
    clue_markup: str,
    enumeration: str,
    answer: str,
    parsing: str,
) -> ParsedClue | None:
    """One ParsedClue from its parts, or None if the answer contradicts the enumeration."""
    clue_markup = _unwrap_whole_clue_color(clue_markup)
    clue_text = strip_markup(clue_markup)
    if not clue_text or len(normalize_answer(answer)) != sum(enumeration_lengths(enumeration)):
        return None  # misread row (or a typo in the post): skip rather than guess
    definitions = _merge_definition_spans(clue_text, _spans(_UNDERLINE, clue_markup))
    return ParsedClue(
        number=number,
        direction=direction,
        clue_markup=clue_markup,
        clue_text=clue_text,
        enumeration=enumeration,
        answer=normalize_answer(answer),
        definitions=definitions,
        indicators=_indicator_spans(clue_markup, clue_text, definitions),
        type_hints=type_hints_from(parsing),
        parsing=strip_markup(parsing) if parsing else "",
        source_url=post.url,
        source_title=post.title,
        setter=_setter_from(post.title),
    )


def _direction_header(text: str) -> Literal["across", "down"] | None:
    """A heading line or table row: 'ACROSS', '**Down**', '|  | **ACROSS** | notes |'.

    One cell must be exactly the word. A row that merely mentions it ("See 20
    Down", "to set down") is not a heading; treating it as one flipped the
    direction of every clue after it.
    """
    cells = _cells(text) if text.lstrip().startswith("|") else [text]
    for cell in cells:
        plain = strip_markup(cell).upper()
        if re.fullmatch(r"\W*ACROSS\W*", plain):
            return "across"
        if re.fullmatch(r"\W*DOWN\W*", plain):
            return "down"
    return None


def parse_clue_table(post: RawPost) -> list[ParsedClue]:
    """Clues laid out as table rows: | number | clue (enum) | answer |, parsing below."""
    clues: list[ParsedClue] = []
    direction: Literal["across", "down"] | None = None
    lines = [line for line in post.content_markdown.splitlines() if line.startswith("|")]

    for i, line in enumerate(lines):
        cells = _cells(line)
        if header := _direction_header(line):
            direction = header
            continue

        number_match = re.match(r"^(\d+)", cells[0]) if cells else None
        if number_match is None:
            continue

        clue_part = answer = parsing = None
        for cell in cells[1:]:
            if clue_part is None and (parsed := _parse_clue_cell(_unwrap_whole_clue_color(cell))):
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

        clue = _make_clue(
            post,
            int(number_match.group(1)),
            direction,
            clue_markup,
            enumeration,
            answer,
            parsing,
        )
        if clue is not None:
            clues.append(clue)
    return clues


# "12 Woo-hoo! Born during ... (7,3)", "18Confront ...", "1. Rubber bird ..." (after any
# whole-clue colour is unwrapped)
_CLUE_LINE = re.compile(r"^\s*(\d+)\.?(?:\s*[,/]\s*\d+)*\s*(?:[ad]\b)?\s*(.+)$", re.I)
# "**WHEELIE BIN**" or "**SANDPAPER** : explanation" (matched with colour tags removed)
_ANSWER_LINE = re.compile(r"^\s*\*\*([A-Z][A-Za-z' \-]*[A-Z])\s*\*\*\s*:?\s*(.*)$")
_COLOR_TAG = re.compile(r"</?color[^>]*>")


def _clue_line(line: str) -> re.Match[str] | None:
    return _CLUE_LINE.match(_unwrap_whole_clue_color(line))


def _answer_line(line: str) -> re.Match[str] | None:
    return _ANSWER_LINE.match(_COLOR_TAG.sub("", line))


def parse_clue_paragraphs(post: RawPost) -> list[ParsedClue]:
    """Clues laid out as paragraphs: clue line, then **ANSWER**, then the explanation.

    1 <u>Starch</u> silly person dropped into sparkling wine (7)
    **CASSAVA**
    ASS (silly person) in CAVA (sparkling wine)
    """
    clues: list[ParsedClue] = []
    direction: Literal["across", "down"] | None = None
    lines = [line for line in post.content_markdown.splitlines() if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if header := _direction_header(line):
            direction = header
            i += 1
            continue
        clue_line = _clue_line(line)
        answer_line = _answer_line(lines[i + 1]) if i + 1 < len(lines) else None
        parsed = (
            _parse_clue_cell(_unwrap_whole_clue_color(clue_line.group(2))) if clue_line else None
        )
        if clue_line is None or answer_line is None or parsed is None:
            i += 1
            continue

        explanation = [answer_line.group(2)] if answer_line.group(2) else []
        j = i + 2
        while j < len(lines) and not _direction_header(lines[j]):
            if _clue_line(lines[j]) and j + 1 < len(lines) and _answer_line(lines[j + 1]):
                break  # the next clue starts
            explanation.append(lines[j])
            j += 1

        clue_markup, enumeration, _ = parsed
        clue = _make_clue(
            post,
            int(clue_line.group(1)),
            direction,
            clue_markup,
            enumeration,
            answer_line.group(1),
            " ".join(explanation),
        )
        if clue is not None:
            clues.append(clue)
        i = j
    return clues


def parse_clues(post: RawPost) -> list[ParsedClue]:
    """Every clue in a post, whichever layout the blogger used."""
    return parse_clue_table(post) or parse_clue_paragraphs(post)
