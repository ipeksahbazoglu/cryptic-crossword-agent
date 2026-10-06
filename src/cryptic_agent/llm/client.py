"""Talking to a chat model, behind a small interface of our own.

The agent depends only on `LLMClient`: send messages (optionally with tools or a
JSON schema), get back a `Completion`. `GroqClient` implements it for Groq's
API; tests use a scripted fake, so they never spend quota or need a network.
Switching provider later means writing one new class.

Messages use the widely shared "OpenAI chat" shape ({"role": ..., "content": ...}),
which Groq and most other providers accept.
"""

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import groq
import httpx

from cryptic_agent import config

logger = logging.getLogger(__name__)

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    """One model reply: text, tool calls, or both."""

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: Usage
    reasoning: str = ""  # the model's private reasoning, when the provider returns it

    def as_message(self) -> Message:
        """The reply as a message to append to the conversation (reasoning left out)."""
        message: Message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        return message


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Completion: ...


@dataclass
class UsageTotals:
    """Running totals, to report what a solve or an evaluation cost."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    waited_seconds: float = 0.0

    def add(self, usage: Usage) -> None:
        self.requests += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens


def parse_duration(text: str) -> float:
    """Groq's reset times: '5.512s' -> 5.512, '1m26.4s' -> 86.4, '250ms' -> 0.25."""
    seconds = 0.0
    for amount, unit in re.findall(r"([\d.]+)(ms|h|m|s)", text):
        seconds += float(amount) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return seconds


@dataclass
class TokenBudget:
    """Pace requests using the budget the API reports after each call.

    Before a request, if the tokens left in the current minute are fewer than a
    typical request needs, wait until the budget resets instead of being refused.
    """

    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    remaining: int | None = None
    resets_at: float = 0.0

    def update(self, headers: Mapping[str, str]) -> None:
        if "x-ratelimit-remaining-tokens" in headers:
            self.remaining = int(headers["x-ratelimit-remaining-tokens"])
            reset = parse_duration(headers.get("x-ratelimit-reset-tokens", "0s"))
            self.resets_at = self.clock() + reset

    def wait_for(self, tokens_needed: int) -> float:
        """Sleep if needed; return how long we waited."""
        if self.remaining is None or self.remaining >= tokens_needed:
            return 0.0
        wait = max(0.0, self.resets_at - self.clock())
        if wait:
            logger.info("token budget low (%d left): waiting %.1fs", self.remaining, wait)
            self.sleep(wait)
        self.remaining = None  # unknown until the next response tells us
        return wait


@dataclass
class GroqClient:
    """LLMClient for Groq's chat completions API."""

    model: str = config.MODEL
    api_key: str | None = None
    expected_tokens: int = 3000  # wait for at least this much budget before a request
    max_retries: int = 5  # the SDK retries 429/5xx with backoff, honouring Retry-After
    http_client: httpx.Client | None = None
    budget: TokenBudget = field(default_factory=TokenBudget)
    totals: UsageTotals = field(default_factory=UsageTotals)

    def __post_init__(self) -> None:
        self._client = groq.Groq(
            api_key=self.api_key or config.get_api_key(),
            max_retries=self.max_retries,
            http_client=self.http_client,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        self.totals.waited_seconds += self.budget.wait_for(self.expected_tokens)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": dict(json_schema)},
            }
        raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        self.budget.update(raw.headers)
        response = raw.parse()

        choice = response.choices[0]
        usage = Usage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        self.totals.add(usage)
        tool_calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in choice.message.tool_calls or []
        ]
        return Completion(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=usage,
            reasoning=getattr(choice.message, "reasoning", None) or "",
        )


def _parse_arguments(arguments: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string; a malformed one becomes {"_raw": ...}."""
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}
    return parsed if isinstance(parsed, dict) else {"_raw": arguments}


@dataclass
class ScriptedLLM:
    """A fake LLMClient for tests: returns prepared completions in order, records requests."""

    replies: list[Completion]
    requests: list[dict[str, Any]] = field(default_factory=list)
    totals: UsageTotals = field(default_factory=UsageTotals)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Completion:
        self.requests.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "json_schema": json_schema}
        )
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        self.totals.add(reply.usage)
        return reply
