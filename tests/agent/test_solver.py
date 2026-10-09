"""The solving loop, driven by a scripted fake model: no network, no quota."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from cryptic_agent.agent.solver import SYSTEM_PROMPT, Solver, Step
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.agent.worksheet import WORKSHEET_SCHEMA, Worksheet
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.llm.client import (
    Completion,
    InvalidToolCallError,
    LLMError,
    RateLimitedError,
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
    return Toolbox(Dictionary(["treason", "senator"]), lexicon)


def test_tool_calls_are_run_and_results_sent_back(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            tool_reply(ToolCall("c1", "find_anagrams", {"letters": "senator"})),
            text_reply("TREASON: an anagram of SENATOR, defined by 'crime'."),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert result.verdict is not None and result.verdict.status == "confirmed"
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c1"
    assert json.loads(tool_message["content"])["matches"] == ["TREASON"]
    assert result.usage == Usage(300, 60)


def test_first_request_has_the_method_and_the_tools(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7", pattern="T??????")

    first = llm.requests[0]
    assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "Senator arranged crime (7)" in first["messages"][1]["content"]
    assert "T??????" in first["messages"][1]["content"]
    assert {t["function"]["name"] for t in first["tools"]} == {*toolbox.tools, "submit_answer"}


def test_worksheet_is_requested_with_the_strict_schema_and_no_tools(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

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

    result = Solver(llm, toolbox, strategy="agent", on_step=seen.append).solve(
        "Senator arranged crime", "7"
    )

    kinds = [s.kind for s in result.steps]
    assert kinds == ["fastpass", "call", "thought", "tool", "call", "thought", "call", "worksheet"]
    assert seen == result.steps
    assert '"reversed":"PARTS"' in result.steps[3].result


def test_fast_pass_evidence_opens_the_conversation(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    first_user_message = llm.requests[0]["messages"][1]["content"]
    assert "Fast pass" in first_user_message
    assert "TREASON" in first_user_message  # anagram of 'Senator', found by code


def test_fast_pass_can_be_switched_off(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply(json.dumps(WORKSHEET))])

    result = Solver(llm, toolbox, strategy="agent", fast_pass=False).solve(
        "Senator arranged crime", "7"
    )

    assert "Fast pass" not in llm.requests[0]["messages"][1]["content"]
    assert "fastpass" not in [s.kind for s in result.steps]


def test_rounds_are_capped(toolbox: Toolbox) -> None:
    looping = [tool_reply(ToolCall(f"c{i}", "reverse_letters", {"text": "x"})) for i in range(2)]
    llm = ScriptedLLM([*looping, text_reply(json.dumps(WORKSHEET))])

    result = Solver(llm, toolbox, strategy="agent", max_rounds=2).solve(
        "Senator arranged crime", "7"
    )

    assert len(llm.requests) == 3  # 2 rounds + the worksheet
    assert llm.requests[-1]["messages"][-2] == {"role": "user", "content": "Stop using tools now."}
    assert result.answer == "TREASON"


def test_an_invalid_worksheet_is_an_error_not_a_crash(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON"), text_reply('{"answer": "TREASON"}')])

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.verdict is None and result.answer is None
    assert "valid worksheet" in result.error


def test_a_failed_worksheet_is_retried(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            text_reply("TREASON"),
            StructuredOutputError("json_validate_failed"),
            text_reply(json.dumps(WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

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

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert [r["effort"] for r in llm.requests[1:]] == ["medium", "medium", "low"]


def test_worksheet_failures_give_an_error_not_a_crash(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply("TREASON")] + [StructuredOutputError("json_validate_failed")] * 3)

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

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

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert "Use only the tools provided" in llm.requests[1]["messages"][-1]["content"]


def test_api_failures_never_crash_a_solve(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([LLMError("RateLimitError: still limited after retries")])

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer is None
    assert "RateLimitError" in result.error


def test_running_out_of_quota_is_raised_not_recorded(toolbox: Toolbox) -> None:
    # Out of quota says nothing about the clue, so it must not become its result.
    llm = ScriptedLLM([RateLimitedError("tokens per day (TPD)")])

    with pytest.raises(RateLimitedError):
        Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")


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


# --- the tiers ---------------------------------------------------------------------


@pytest.fixture
def tier_toolbox(lexicon: Lexicon) -> Toolbox:
    # The miniature lexicon knows "Love god" -> EROS and "sparkling" as an anagram
    # indicator, so the fast pass finds EROS from both sides.
    return Toolbox(Dictionary(["eros", "ores", "roes", "sore", "rose"]), lexicon)


EROS_WORKSHEET: dict[str, Any] = {
    "answer": "EROS",
    "definitions": [{"text": "Love god's", "position": "start"}],
    "wordplay": [
        {
            "mechanism": "anagram",
            "indicator": "sparkling",
            "fodder": "rose",
            "produces": "EROS",
            "explanation": "ROSE rearranged",
        }
    ],
    "link_words": [],
    "confidence": "confirmed",
    "alternatives": [],
}


def test_tier_0_solves_with_code_alone(tier_toolbox: Toolbox) -> None:
    llm = ScriptedLLM([])  # any model call would fail: there are no replies

    result = Solver(llm, tier_toolbox).solve("Love god's sparkling rose", "4")

    assert (result.answer, result.tier) == ("EROS", 0)
    assert result.verdict is not None and result.verdict.status == "confirmed"
    assert result.usage == Usage(0, 0)
    assert llm.requests == []


def test_an_answer_the_definition_does_not_support_is_never_confirmed(
    tier_toolbox: Toolbox,
) -> None:
    # Crossing letters rule EROS out. ORES is a real anagram of "rose", but so are
    # ROES and SORE, and nothing says "Love god" means ORES: every tier must refuse it.
    ores = copy.deepcopy(EROS_WORKSHEET)
    ores["answer"] = ores["wordplay"][0]["produces"] = "ORES"
    llm = ScriptedLLM(
        [text_reply(json.dumps(ores)), tool_reply(ToolCall("s1", "submit_answer", ores))]
    )

    result = Solver(llm, tier_toolbox).solve("Love god's sparkling rose", "4", pattern="O???")

    assert result.tier == 2
    assert result.verdict is not None and result.verdict.status == "pencilled"
    meaning = next(c for c in result.verdict.checks if c.name == "definition means the answer")
    assert meaning.passed is None and "ROES" in meaning.detail


def test_tier_1_is_one_call_without_tools(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([text_reply(json.dumps(WORKSHEET))])

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert (result.answer, result.tier) == ("TREASON", 1)
    assert len(llm.requests) == 1
    assert llm.requests[0]["tools"] is None
    assert llm.requests[0]["json_schema"] == WORKSHEET_SCHEMA
    assert "Fast pass" in llm.requests[0]["messages"][1]["content"]


def test_unconfirmed_tier_1_escalates_with_what_went_wrong(toolbox: Toolbox) -> None:
    wrong = WORKSHEET | {"definitions": [{"text": "crime", "position": "start"}]}
    llm = ScriptedLLM(
        [
            text_reply(json.dumps(wrong)),  # tier 1: definition not where it claims
            tool_reply(ToolCall("s1", "submit_answer", WORKSHEET)),  # tier 2 submits
        ]
    )

    result = Solver(llm, toolbox).solve("Senator arranged crime", "7")

    assert (result.answer, result.tier) == ("TREASON", 2)
    tier_2_prompt = llm.requests[1]["messages"][1]["content"]
    assert "quick first attempt answered TREASON" in tier_2_prompt
    assert "not at the start" in tier_2_prompt


def test_submit_answer_ends_the_loop_without_a_worksheet_call(toolbox: Toolbox) -> None:
    llm = ScriptedLLM([tool_reply(ToolCall("s1", "submit_answer", WORKSHEET))])

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert len(llm.requests) == 1  # no separate worksheet round trip


def test_an_invalid_submission_is_sent_back_to_fix(toolbox: Toolbox) -> None:
    llm = ScriptedLLM(
        [
            tool_reply(ToolCall("s1", "submit_answer", {"answer": "TREASON"})),
            tool_reply(ToolCall("s2", "submit_answer", WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    assert result.answer == "TREASON"
    assert "fix and resubmit" in llm.requests[1]["messages"][-1]["content"]


def test_tier_1_is_skipped_when_code_found_no_wordplay(toolbox: Toolbox) -> None:
    # Nothing in "Quiet dog barks" anagrams, hides or spells a 5-letter word here,
    # so a one-shot call would have nothing to build on: go straight to the agent.
    llm = ScriptedLLM([tool_reply(ToolCall("s1", "submit_answer", WORKSHEET))])

    result = Solver(llm, toolbox).solve("Quiet dog barks", "5")

    assert result.tier == 2
    assert len(llm.requests) == 1 and llm.requests[0]["tools"] is not None
    assert [s.text for s in result.steps if s.kind == "tier"] == [
        "tier 0: code only",
        "tier 2: the agent with tools",
    ]


def test_a_repeated_tool_call_is_answered_with_a_nudge_not_run_again(toolbox: Toolbox) -> None:
    same = {"text": "strap"}
    llm = ScriptedLLM(
        [
            tool_reply(ToolCall("c1", "reverse_letters", same)),
            tool_reply(ToolCall("c2", "reverse_letters", same)),
            tool_reply(ToolCall("s1", "submit_answer", WORKSHEET)),
        ]
    )

    result = Solver(llm, toolbox, strategy="agent").solve("Senator arranged crime", "7")

    first, second = [s.result for s in result.steps if s.kind == "tool"]
    assert "PARTS" in first
    assert "already made exactly this call" in second and "PARTS" not in second
