"""The solving loop, driven by a scripted fake model: no network, no quota."""

import json
from pathlib import Path

import pytest

from cryptic_agent.agent.solver import SYSTEM_PROMPT, Solver, Step
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.agent.worksheet import WORKSHEET_SCHEMA, Worksheet
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.llm.client import (
    Completion,
    InvalidToolCallError,
    LLMError,
    ScriptedLLM,
    StructuredOutputError,
    ToolCall,
    Usage,
)
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


def tool_reply(*calls: ToolCall) -> Completion:
    return Completion("", list(calls), "tool_calls", Usage(100, 20), reasoning="thinking")


def text_reply(text: str) -> Completion:
    return Completion(text, [], "stop", Usage(100, 20))


@pytest.fixture
def toolbox(lexicon: Lexicon) -> Toolbox:
    return Toolbox(Dictionary(["treason", "senator", "atoners"]), lexicon)


def test_tool_calls_are_run_and_results_sent_back(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            tool_reply(ToolCall("c1", "find_anagrams", {"letters": "senator"})),
            text_reply("TREASON: an anagram of SENATOR, defined by 'crime'."),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert result.verdict is not None and result.verdict.status == "confirmed"
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c1"
    assert json.loads(tool_message["content"])["matches"] == ["ATONERS", "TREASON"]
    assert result.usage == Usage(300, 60)


def test_first_request_has_the_method_and_the_tools(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    Solver(llm, toolbox).solve("Senator arranged crime", "7", pattern="T??????")

    first = llm.requests[0]
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Senator arranged crime (7)" in first["messages"][1]["content"]
    assert "T??????" in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == set(toolbox.tools)


def test_worksheet_is_requested_with_the_strict_schema_and_no_tools(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    Solver(llm, toolbox).solve("Senator arranged crime", "7")

    final = llm.requests[-1]
    assert final["json_schema"] == WORKSHEET_SCHEMA
    assert final["tools"] is None


def test_steps_are_recorded_and_streamed(toolbox: Toolbox) -> None:
    seen: list[Step] = []
    llm = ScriptedLLM(
        [
            tool_reply(ToolCall("c1", "reverse_letters", {"text": "strap"})),
            text_reply("done"),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox, on_step=seen.append).solve("Senator arranged crime", "7")

    assert [s.kind for s in result.steps] == ["thought", "tool", "thought", "worksheet"]
    assert seen == result.steps
    assert '"reversed":"PARTS"' in result.steps[1].result


def test_rounds_are_capped(toolbox: Toolbox) -> None:
    looping = [tool_reply(ToolCall(f"c{i}", "reverse_letters", {"text": "x"})) for i in range(2)]
    llm = ScriptedLLM([*looping, text_reply(json.dumps(WORKSHEET))])

    result = Solver(llm, toolbox, max_rounds=2).solve("Senator arranged crime", "7")

    assert len(llm.requests) == 3  # 2 rounds + the worksheet
    assert llm.requests[-1]["messages"][-2] == {"role": "user", "content": "Stop using tools now."}
    assert result.answer == "TREASON"


def test_an_invalid_worksheet_is_an_error_not_a_crash(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply('{"answer": "TREASON"}')])

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.verdict is None and result.answer is None
    assert "definitions" in result.error


def test_a_failed_worksheet_is_retried(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            text_reply("TREASON"),
            StructuredOutputError("json_validate_failed"),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.verdict is not None and result.verdict.status == "confirmed"
    assert "ONLY the JSON worksheet" in llm.requests[-1]["messages"][-1]["content"]
    assert [r["effort"] for r in llm.requests[-2:]] == ["medium", "medium"]


def test_low_effort_is_the_last_resort(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            text_reply("TREASON"),
            StructuredOutputError("json_validate_failed"),
            InvalidToolCallError("Tool choice is none, but model called a tool"),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert [r["effort"] for r in llm.requests[1:]] == ["medium", "medium", "low"]


def test_worksheet_failures_give_an_error_not_a_crash(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON")] + [StructuredOutputError("json_validate_failed")] * 3)

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.verdict is None
    assert "valid worksheet" in result.error


def test_an_invalid_tool_call_mid_solve_is_recovered(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            InvalidToolCallError("attempted to call tool 'commentary'"),
            text_reply("TREASON"),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert "Use only the tools provided" in llm.requests[1]["messages"][-1]["content"]


def test_api_failures_never_crash_a_solve(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([LLMError("RateLimitError: still limited after retries")])

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert result.answer is None
    assert "RateLimitError" in result.error


def test_worksheet_schema_matches_the_model() -> None:
    # The hand-written strict schema and the pydantic model must describe the same fields.
    assert set(WORKSHEET_SCHEMA["properties"]) == set(Worksheet.model_fields)
    step_schema = WORKSHEET_SCHEMA["properties"]["wordplay"]["items"]
    assert set(step_schema["properties"]) == set(
        Worksheet.model_fields["wordplay"].annotation.__args__[0].model_fields  # type: ignore[union-attr]
    )

    def strict(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                strict(value)

    strict(WORKSHEET_SCHEMA)
    Worksheet.model_validate(WORKSHEET)  # the example used above is itself valid


def test_lexicon_fixture_path_exists(lexicon_path: Path) -> None:
    assert lexicon_path.exists()
