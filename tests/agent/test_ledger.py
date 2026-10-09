import json

import pytest

from cryptic_agent.agent.ledger import format_ledger, ledger, request_overhead
from cryptic_agent.agent.solver import Solver
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.llm.client import Completion, ScriptedLLM, ToolCall, Usage
from cryptic_agent.tools.dictionary import Dictionary

WORKSHEET = {
    "answer": "TREASON",
    "definitions": [{"text": "crime", "position": "end"}],
    "wordplay": [
        {
            "mechanism": "anagram",
            "indicator": "arranged",
            "fodder": "Senator",
            "produces": "TREASON",
            "explanation": "SENATOR rearranged",
        }
    ],
    "link_words": [],
    "confidence": "confirmed",
    "alternatives": [],
}


@pytest.fixture
def toolbox(lexicon: Lexicon) -> Toolbox:
    return Toolbox(Dictionary(["treason", "senator"]), lexicon)


def test_ledger_has_one_row_per_model_call_and_adds_up(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            Completion(
                "",
                [ToolCall("c1", "find_anagrams", {"letters": "senator"})],
                "tool_calls",
                Usage(prompt_tokens=1000, completion_tokens=500, reasoning_tokens=460),
                reasoning="thinking",
            ),
            Completion(
                "",
                [ToolCall("s1", "submit_answer", WORKSHEET)],
                "tool_calls",
                Usage(prompt_tokens=1200, completion_tokens=300, reasoning_tokens=200),
            ),
        ]
    )

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")
    rows = ledger(result)

    assert [r.step for r in rows] == ["tier 2: round 1", "tier 2: round 2"]
    assert (rows[0].sent, rows[0].reasoning, rows[0].answer) == (1000, 460, 40)
    assert rows[0].did == "find_anagrams"
    assert rows[1].did == "handed in the worksheet"
    assert [r.running_total for r in rows] == [1500, 3000]
    assert rows[-1].running_total == result.usage.total_tokens
    assert result.usage.reasoning_tokens == 660


def test_code_only_solve_has_an_empty_ledger(lexicon: Lexicon) -> None:
    toolbox = Toolbox(Dictionary(["eros", "ores", "roes", "sore", "rose"]), lexicon)

    result = Solver(ScriptedLLM([]), toolbox).solve("Love god's sparkling rose", "4")

    assert ledger(result) == []
    assert "0 tokens" in format_ledger(result)


def test_formatted_ledger_shows_every_call_and_the_total(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([Completion(json.dumps(WORKSHEET), [], "stop", Usage(900, 3600, 3500))])

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")
    text = format_ledger(result)

    assert "tier 1: worksheet (medium effort)" in text
    assert "3,500" in text  # reasoning
    assert "4,500" in text  # total


def test_request_overhead_counts_instructions_and_tools(toolbox: Toolbox) -> None:
    overhead = request_overhead(toolbox)

    assert set(overhead) == {"instructions", "tool descriptions"}
    assert overhead["tool descriptions"] > overhead["instructions"] > 0
