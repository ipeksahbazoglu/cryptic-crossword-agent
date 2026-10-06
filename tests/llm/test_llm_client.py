"""GroqClient tests run against a fake HTTP transport: real SDK, no network, no quota."""

import json
from typing import Any

import httpx
import pytest

from cryptic_agent.llm.client import (
    Completion,
    GroqClient,
    InvalidToolCallError,
    LLMError,
    ScriptedLLM,
    StructuredOutputError,
    TokenBudget,
    ToolCall,
    Usage,
)

RATE_HEADERS = {
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-tokens": "7265",
    "x-ratelimit-reset-tokens": "5.5s",
}


def chat_response(message: dict[str, Any], finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 151, "completion_tokens": 236, "total_tokens": 387},
    }


class FakeGroq:
    """Records requests and replies with prepared responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return self.responses.pop(0)


def make_client(fake: FakeGroq, budget: TokenBudget | None = None) -> GroqClient:
    return GroqClient(
        api_key="gsk-test",
        http_client=httpx.Client(transport=httpx.MockTransport(fake)),
        budget=budget or TokenBudget(),
    )


# --- parsing replies ------------------------------------------------------------


def test_tool_calls_are_parsed() -> None:
    fake = FakeGroq(
        httpx.Response(
            200,
            headers=RATE_HEADERS,
            json=chat_response(
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning": "Senator has 7 letters; 'arranged' suggests an anagram.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "find_anagrams",
                                "arguments": '{"letters": "senator"}',
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        )
    )

    reply = make_client(fake).complete(
        [{"role": "user", "content": "Senator arranged crime (7)"}],
        tools=[{"type": "function", "function": {"name": "find_anagrams", "parameters": {}}}],
    )

    assert reply.tool_calls == [ToolCall("call_1", "find_anagrams", {"letters": "senator"})]
    assert reply.finish_reason == "tool_calls"
    assert reply.reasoning.startswith("Senator has 7 letters")
    assert reply.usage == Usage(151, 236)
    assert fake.requests[0]["model"] == "openai/gpt-oss-120b"
    assert "tools" in fake.requests[0]


def test_json_schema_is_requested_strictly() -> None:
    fake = FakeGroq(
        httpx.Response(200, json=chat_response({"role": "assistant", "content": '{"a": 1}'}))
    )
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}

    reply = make_client(fake).complete([{"role": "user", "content": "hi"}], json_schema=schema)

    assert reply.content == '{"a": 1}'
    response_format = fake.requests[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == schema


def test_schema_validation_failure_becomes_structured_output_error() -> None:
    # What Groq returns when strict-schema output doesn't validate (seen live).
    error = {
        "error": {
            "message": "Failed to validate JSON.",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "",
        }
    }
    fake = FakeGroq(httpx.Response(400, json=error))

    with pytest.raises(StructuredOutputError, match="json_validate_failed"):
        make_client(fake).complete([{"role": "user", "content": "x"}], json_schema={})


def test_invalid_tool_call_is_recognised() -> None:
    # Seen live: gpt-oss leaking its internal "commentary" channel as a tool call.
    error = {
        "error": {
            "message": "attempted to call tool 'commentary' which was not in request.tools",
            "type": "invalid_request_error",
            "code": "tool_use_failed",
        }
    }
    fake = FakeGroq(httpx.Response(400, json=error))

    with pytest.raises(InvalidToolCallError, match="commentary"):
        make_client(fake).complete([{"role": "user", "content": "x"}])


def test_other_api_errors_become_llm_errors() -> None:
    fake = FakeGroq(httpx.Response(400, json={"error": {"message": "bad model"}}))

    with pytest.raises(LLMError, match="bad model") as info:
        make_client(fake).complete([{"role": "user", "content": "x"}])
    assert not isinstance(info.value, (StructuredOutputError, InvalidToolCallError))


def test_reasoning_effort_is_sent_when_asked() -> None:
    fake = FakeGroq(
        *[httpx.Response(200, json=chat_response({"role": "assistant"})) for _ in range(2)]
    )
    client = make_client(fake)

    client.complete([{"role": "user", "content": "x"}], effort="low")
    client.complete([{"role": "user", "content": "x"}])

    assert fake.requests[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in fake.requests[1]


def test_malformed_tool_arguments_are_kept_raw() -> None:
    fake = FakeGroq(
        httpx.Response(
            200,
            json=chat_response(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        )
    )

    reply = make_client(fake).complete([{"role": "user", "content": "x"}])

    assert reply.tool_calls[0].arguments == {"_raw": "{not json"}


def test_reply_becomes_a_conversation_message() -> None:
    reply = Completion(
        content="",
        tool_calls=[ToolCall("call_1", "find_anagrams", {"letters": "senator"})],
        finish_reason="tool_calls",
        usage=Usage(),
        reasoning="private",
    )

    message = reply.as_message()

    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["function"] == {
        "name": "find_anagrams",
        "arguments": '{"letters": "senator"}',
    }
    assert "private" not in json.dumps(message)  # reasoning is never sent back


# --- rate limits and usage --------------------------------------------------------


def test_usage_is_totalled_and_budget_read_from_headers() -> None:
    fake = FakeGroq(
        *[
            httpx.Response(200, headers=RATE_HEADERS, json=chat_response({"role": "assistant"}))
            for _ in range(2)
        ]
    )
    client = make_client(fake)

    client.complete([{"role": "user", "content": "a"}])
    client.complete([{"role": "user", "content": "b"}])

    assert (client.totals.requests, client.totals.prompt_tokens) == (2, 302)
    assert client.budget.remaining == 7265


def test_rate_limited_request_is_retried() -> None:
    fake = FakeGroq(
        httpx.Response(429, headers={"retry-after-ms": "1"}, json={"error": {"message": "slow"}}),
        httpx.Response(200, json=chat_response({"role": "assistant", "content": "TREASON"})),
    )

    reply = make_client(fake).complete([{"role": "user", "content": "x"}])

    assert reply.content == "TREASON"
    assert len(fake.requests) == 2


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def budget_with(remaining: int, clock: FakeClock) -> TokenBudget:
    budget = TokenBudget(clock=clock, sleep=clock.sleep)
    budget.update(
        {"x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": str(remaining)}
    )
    return budget


def test_budget_waits_only_for_the_refill_it_needs() -> None:
    clock = FakeClock()
    budget = budget_with(1000, clock)

    waited = budget.wait_for(3000)  # 2,000 short at 8,000/min = 133.3/s -> 15 s

    assert waited == pytest.approx(15.0)
    assert clock.sleeps == [pytest.approx(15.0)]


def test_budget_counts_refill_since_the_last_response() -> None:
    clock = FakeClock()
    budget = budget_with(1000, clock)
    clock.now += 9  # 9 s later, 1,200 tokens have refilled

    assert budget.wait_for(3000) == pytest.approx(6.0)


def test_budget_does_not_wait_when_enough_or_unknown() -> None:
    clock = FakeClock()

    assert TokenBudget(clock=clock, sleep=clock.sleep).wait_for(3000) == 0.0  # nothing known
    assert budget_with(7000, clock).wait_for(3000) == 0.0
    assert clock.sleeps == []


# --- the fake used by agent tests ---------------------------------------------------


def test_scripted_llm_replays_and_records() -> None:
    llm = ScriptedLLM([Completion("TREASON", [], "stop", Usage(10, 5))])

    reply = llm.complete([{"role": "user", "content": "Senator arranged crime (7)"}])

    assert reply.content == "TREASON"
    assert llm.requests[0]["messages"][0]["content"] == "Senator arranged crime (7)"
    assert llm.totals.completion_tokens == 5
    with pytest.raises(AssertionError, match="ran out"):
        llm.complete([])
