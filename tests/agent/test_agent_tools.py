import json
from typing import Any

import pytest

from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.tools.dictionary import Dictionary


@pytest.fixture
def toolbox(lexicon: Lexicon) -> Toolbox:
    return Toolbox(Dictionary(["treason", "senator", "atoners", "scrum", "parts"]), lexicon)


def run(toolbox: Toolbox, name: str, **arguments: object) -> dict[str, Any]:
    """A tool's result, parsed from the JSON the model would see."""
    result: dict[str, Any] = json.loads(toolbox.run(name, dict(arguments)))
    return result


def test_every_declared_tool_has_matching_parameters(toolbox: Toolbox) -> None:
    for spec in toolbox.specs():
        function = spec["function"]
        assert function["name"] in toolbox.tools
        assert function["description"]
        params = function["parameters"]
        assert set(params["required"]) <= set(params["properties"])


def test_wordplay_tools(toolbox: Toolbox) -> None:
    assert run(toolbox, "find_anagrams", letters="senator")["matches"] == ["ATONERS", "TREASON"]
    assert run(toolbox, "reverse_letters", text="strap")["reversed"] == "PARTS"
    assert (
        run(toolbox, "select_letters", text="cleric over day explained", which="first")["letters"]
        == "CODE"
    )
    assert run(toolbox, "check_answer", answer="treason", enumeration="(7)")["valid"] is True


def test_hidden_word_can_search_backwards(toolbox: Toolbox) -> None:
    forwards = run(toolbox, "check_hidden_word", text="delicious crumble", length=5)
    backwards = run(toolbox, "check_hidden_word", text="seas trap", length=5, reversed=True)

    assert [m["word"] for m in forwards["matches"]] == ["SCRUM"]
    assert [m["word"] for m in backwards["matches"]] == ["PARTS"]


def test_lexicon_tools_report_support(toolbox: Toolbox) -> None:
    result = run(toolbox, "definition_answers", phrase="love god", length=4)

    assert result == {"results": [{"value": "EROS", "clues": 3}], "total": 1}
    assert {"value": "I", "clues": 0, "curated": True} in run(
        toolbox, "abbreviations", phrase="one"
    )["results"]


def test_excluded_puzzles_reach_the_tools(lexicon: Lexicon, guardian_urls: list[str]) -> None:
    toolbox = Toolbox(Dictionary(["eros"]), lexicon, exclude_urls=guardian_urls)

    assert run(toolbox, "definition_answers", phrase="love god")["results"] == [
        {"value": "EROS", "clues": 1}
    ]


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("solve_it_for_me", {}, "unknown tool"),
        ("find_anagrams", {"_raw": "{bad"}, "not valid JSON"),
        ("find_anagrams", {"wrong": "x"}, "TypeError"),
        ("check_hidden_word", {"text": "abc", "length": 0}, "ValueError"),
        ("find_anagrams", {"letters": 123}, "AttributeError"),  # a number where text belongs
        ("definition_answers", {"phrase": None}, "AttributeError"),
    ],
)
def test_problems_come_back_as_errors_the_model_can_read(
    toolbox: Toolbox, name: str, arguments: dict[str, object], message: str
) -> None:
    result = json.loads(toolbox.run(name, arguments))

    assert message in result["error"]
