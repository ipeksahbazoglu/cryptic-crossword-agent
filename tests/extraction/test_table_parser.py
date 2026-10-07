"""Table parser tests, built from real rows of each layout Fifteensquared has used."""

from datetime import datetime

import pytest

from cryptic_agent.extraction.table_parser import (
    parse_clue_table,
    parse_clues,
    strip_markup,
    type_hints_from,
)
from cryptic_agent.models import RawPost


def post(table: str, title: str = "Guardian Quick Cryptic 129 by Chandler") -> RawPost:
    return RawPost(
        id=1,
        date=datetime(2026, 9, 19),
        url="https://fifteensquared.net/qc-129/",
        title=title,
        content_markdown="Intro text.\n\n|  |  |  |\n| --- | --- | --- |\n" + table,
    )


# 2025-03 onwards: "Answer X" cell, parsing on the following row.
CURRENT = """\
|  | **ACROSS** | Click on “Answer” to see the solutions |
| 1 | <color=red>After spending money</color>, vigorously bite <u>**small cut of meat**</u> (4) | Answer CHOP |
|  | Parsing *deletion* (after spending money) from CHO[m]P (vigorously bite) |  |
| 7 | <color=red>Cook</color> using mere <u>**sweet desserts**</u> (9) | Answer MERINGUES |
|  | Parsing *anagram* of (USING MERE)\\* with an *anagrind* of “cook” |  |
|  | **DOWN** |  |
| 6 | <u>**Outside of**</u> <u>**pub**</u>? (3) | Answer BAR |
|  | Parsing *double definition* – think everything but (outside of). |  |
"""

# Late 2024: bare answer cell.
LATE_2024 = """\
|  | **ACROSS** | Click on “details” to see the solutions |
| 1 | <u>**Toddlers’ facilities**</u>, most blotchy, <color=red>out of bounds</color> (7) | POTTIES |
|  | *naked word* (out of bounds) from sPOTTIESt (most blotchy) |  |
"""

# Early 2024: answer first; clue, enumeration and parsing share one cell.
EARLY_2024 = """\
|  | ACROSS |  |
| 1 | CONSIDERATE | <u>**Kind** </u>of awful desecration (11)  *anagram* (DESECRATION)\\* – *anagrind* = awful |
| 8 | EDGE | <u>**Limit**</u> unwarranted generosity in part (4)  *hidden* (in part) in unwarrent**ED GE**nerosity |
"""


def test_current_layout() -> None:
    chop, meringues, bar = parse_clue_table(post(CURRENT))

    assert chop.clue_text == "After spending money, vigorously bite small cut of meat"
    assert (chop.number, chop.direction, chop.enumeration, chop.answer) == (
        1,
        "across",
        "4",
        "CHOP",
    )
    assert chop.definitions == ["small cut of meat"]
    assert chop.indicators == ["After spending money"]
    assert chop.type_hints == ["deletion"]
    assert chop.parsing.startswith("deletion (after spending money)")
    assert chop.setter == "Chandler"

    assert meringues.type_hints == ["anagram"]  # "anagrind" is not a type
    assert "(USING MERE)*" in meringues.parsing  # anagram asterisk unescaped

    assert bar.direction == "down"
    assert bar.definitions == ["Outside of", "pub"]  # one at each end: kept apart
    assert bar.type_hints == ["double_definition"]


def test_late_2024_layout_with_bare_answer() -> None:
    (potties,) = parse_clue_table(post(LATE_2024, title="Quick Cryptic 47 by Ludwig"))

    assert potties.answer == "POTTIES"
    assert potties.definitions == ["Toddlers’ facilities"]
    assert potties.indicators == ["out of bounds"]
    assert potties.type_hints == ["deletion"]  # "naked word"
    assert potties.setter == "Ludwig"


def test_early_2024_layout_with_parsing_in_clue_cell() -> None:
    considerate, edge = parse_clue_table(
        post(EARLY_2024, title="Guardian Quick Cryptic 1 – Carpathian")
    )

    assert considerate.clue_text == "Kind of awful desecration"
    assert considerate.answer == "CONSIDERATE"
    assert considerate.definitions == ["Kind"]
    assert considerate.indicators == []  # this era did not colour indicators
    assert considerate.parsing.startswith("anagram (DESECRATION)*")
    assert edge.type_hints == ["hidden_word"]
    assert considerate.setter == "Carpathian"


def test_answer_length_mismatch_is_skipped() -> None:
    # A real blogger typo: MALAWI has 6 letters, the post says (7).
    table = "| 13 | MALAWI | Mother to rule one<u> **country**</u> (7) *charade* MA + LAW + I |\n"

    assert parse_clue_table(post(table)) == []


def test_mixed_case_answer_is_normalized() -> None:
    # Bloggers lowercase the letter a wordplay changes: PiER.
    table = "| 6 | One playing <color=red>without piano</color> in <u>**part of resort**</u> (4) | Answer PiER |\n"

    assert parse_clue_table(post(table))[0].answer == "PIER"


def test_word_by_word_underlining_is_merged() -> None:
    table = "| 3 | <u>**Toddlers’**</u> <u>**facilities**</u> most blotchy (7) | POTTIES |\n"

    assert parse_clue_table(post(table))[0].definitions == ["Toddlers’ facilities"]


def test_enumeration_in_the_parsing_is_not_mistaken_for_the_clues() -> None:
    # Early layout: the parsing shares the cell and can mention another enumeration.
    table = (
        "| 16 | TEST | **<u>Try</u>** selection of chocolate stores (4) "
        "*hidden* in chocola**TE ST**ores, as in 9 (5) |\n"
    )

    (test,) = parse_clue_table(post(table))

    assert (test.clue_text, test.enumeration) == ("Try selection of chocolate stores", "4")
    assert test.parsing.endswith("as in 9 (5)")


@pytest.mark.parametrize(
    "row",
    [
        "| 15 | See 3 | RING |",  # cross-reference, no clue
        "| 28 | Criminal leads <u>**trades**</u> (5) |  |",  # no answer cell
        "|  | Parsing *anagram* of FOO |  |",  # not a clue row
    ],
)
def test_rows_without_a_full_clue_are_ignored(row: str) -> None:
    assert parse_clue_table(post(row + "\n")) == []


@pytest.mark.parametrize(
    ("parsing", "hints"),
    [
        ("*anagram of* (CRUET)\\* with *anagrind* of “mixed”", ["anagram"]),
        ("*soundalike of* (you say) of “queuing”", ["homophone"]),
        ("*charade* of BA + KING, then *reversal*", ["charade", "reversal"]),
        ("**bold** is not italic, *nothing to map*", []),
    ],
)
def test_type_hints_from_italics(parsing: str, hints: list[str]) -> None:
    assert type_hints_from(parsing) == hints


def test_strip_markup() -> None:
    markup = "<color=red>Cook</color> using\xa0mere <u>**sweet desserts**</u> (USING MERE)\\*"

    assert strip_markup(markup) == "Cook using mere sweet desserts (USING MERE)*"


# --- paragraph layouts used by many full Guardian cryptic bloggers -------------------

PARAGRAPHS = """\
Definitions are underlined in the clues.

**Across**

1 <color=#0000ff><u>Starch</u> silly person dropped into sparkling wine (7)</color>
**CASSAVA**
ASS (silly person) in CAVA (sparkling wine)

18<color=#0000ff> <u>Confront</u> a constant expense (6)</color>
**ACCOST**
A + C (constant) + COST (expense)

**Down**

16 <color=#0000ff>A small train diverted to <u>Nancy?</u> (7)</color>
**SINATRA**
An anagram (diverted) of A S[mall] TRAIN – neat misdirection
"""

# Colour tags around the number and the answer; explanation after a colon.
COLOURED_PARAGRAPHS = """\
**Across**
<color=#0000ff>1. <u>Rubber</u> bird I replaced by middle of May (9)</color>
<color=#ff0000>**SANDPAPER** </color>: "sandpiper" with "I" replacing middle letter of "May".

<color=#0000ff>6. <u>Best</u> comedian with shaved head (5)</color>
<color=#ff0000>**CREAM** </color>: "scream" minus its 1st letter.
"""


def paragraph_post(body: str) -> RawPost:
    return RawPost(
        id=2,
        date=datetime(2026, 9, 30),
        url="https://fifteensquared.net/guardian-30125-imogen/",
        title="Guardian 30,125 / Imogen",
        content_markdown=body,
    )


def test_paragraph_layout() -> None:
    cassava, accost, sinatra = parse_clues(paragraph_post(PARAGRAPHS))

    assert (cassava.number, cassava.direction, cassava.answer) == (1, "across", "CASSAVA")
    assert cassava.clue_text == "Starch silly person dropped into sparkling wine"
    assert cassava.definitions == ["Starch"]
    assert cassava.indicators == []  # the blue wraps the whole clue: not an indicator
    assert cassava.parsing == "ASS (silly person) in CAVA (sparkling wine)"
    assert accost.number == 18  # "18<color..." with no space
    assert (sinatra.direction, sinatra.type_hints) == ("down", ["anagram"])
    assert sinatra.definitions == ["Nancy"]


def test_coloured_paragraph_layout() -> None:
    sandpaper, cream = parse_clues(paragraph_post(COLOURED_PARAGRAPHS))

    assert (sandpaper.number, sandpaper.answer, sandpaper.definitions) == (
        1,
        "SANDPAPER",
        ["Rubber"],
    )
    assert sandpaper.parsing.startswith('"sandpiper"')
    assert cream.parsing == '"scream" minus its 1st letter.'  # not run into the next clue


def test_tables_are_preferred_when_present() -> None:
    assert [c.answer for c in parse_clues(post(CURRENT))] == ["CHOP", "MERINGUES", "BAR"]


@pytest.mark.parametrize(
    "markup",
    [
        "<color=#ff0000><u>One whose number is up, having rounded Lido recklessly?</u></color>",
        "<color=#4682b4><u>This marine</u> is deeply blue</color>",  # colour wraps a definition
    ],
)
def test_colour_on_a_whole_clue_or_definition_is_not_an_indicator(markup: str) -> None:
    table = f"| 1 | {markup} (8) | Answer DEADBEAT |\n"

    (clue,) = parse_clue_table(post(table))

    assert clue.indicators == []


@pytest.mark.parametrize(
    ("parsing", "hints"),
    [
        ("An anagram (diverted) of A S[mall] TRAIN", ["anagram"]),
        ("Double definition (for the first, Chambers has ...)", ["double_definition"]),
        ("A reversal (turning) of MOOD (state of mind)", ["reversal"]),
        ("Hidden in tHE ROse", ["hidden_word"]),
        ("ASS (silly person) in CAVA; the anagram idea is a red herring", []),  # aside
    ],
)
def test_type_hints_from_plain_opening_words(parsing: str, hints: list[str]) -> None:
    assert type_hints_from(parsing) == hints
