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
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import groq
import httpx

from cryptic_agent import config

logger = logging.getLogger(__name__)

Message = dict[str, Any]
Effort = Literal["low", "medium", "high"]  # how much a reasoning model thinks first


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


class LLMError(RuntimeError):
    """Anything that went wrong talking to the model. Callers handle only these."""


class StructuredOutputError(LLMError):
    """The model did not produce output matching the requested JSON schema."""


class InvalidToolCallError(LLMError):
    """The model called a tool that doesn't exist, so the provider rejected the reply.

    A mistake the model can recover from: tell it and let it continue.
    """


class RateLimitedError(LLMError):
    """Out of quota even after retries (e.g. the free tier's 200,000 tokens per day).

    Not a problem with any one clue: callers should stop and resume later.
    """


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
        effort: Effort | None = None,
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


@dataclass
class TokenBudget:
    """Pace requests using the budget the API reports after each call.

    The per-minute budget refills continuously (8,000 tokens/min is about 133 a
    second), so before a request we wait only as long as it takes to refill
    enough for a typical request, not for the whole budget to reset.
    """

    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    remaining: int | None = None
    per_minute: int | None = None
    seen_at: float = 0.0

    def update(self, headers: Mapping[str, str]) -> None:
        if "x-ratelimit-remaining-tokens" in headers:
            self.remaining = int(headers["x-ratelimit-remaining-tokens"])
            self.seen_at = self.clock()
        if "x-ratelimit-limit-tokens" in headers:
            self.per_minute = int(headers["x-ratelimit-limit-tokens"])

    def wait_for(self, tokens_needed: int) -> float:
        """Sleep until about `tokens_needed` tokens are available; return the wait."""
        if self.remaining is None or not self.per_minute:
            return 0.0
        refill_per_second = self.per_minute / 60
        available = self.remaining + (self.clock() - self.seen_at) * refill_per_second
        wait = (tokens_needed - available) / refill_per_second
        if wait <= 0:
            return 0.0
        logger.info("token budget low (~%d left): waiting %.1fs", available, wait)
        self.sleep(wait)
        return wait


@dataclass
class GroqClient:
    """LLMClient for Groq's chat completions API."""

    model: str = config.MODEL
    api_key: str | None = None
    expected_tokens: int = 2500  # a typical request: ~1k prompt + ~1.5k output incl. reasoning
    max_retries: int = 5  # the SDK retries 429/5xx with backoff, honouring Retry-After
    timeout: float = 60.0  # seconds per attempt, so a stuck connection fails fast and retries
    http_client: httpx.Client | None = None
    budget: TokenBudget = field(default_factory=TokenBudget)
    totals: UsageTotals = field(default_factory=UsageTotals)

    def __post_init__(self) -> None:
        self._client = groq.Groq(
            api_key=self.api_key or config.get_api_key(),
            max_retries=self.max_retries,
            timeout=self.timeout,
            http_client=self.http_client,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
        effort: Effort | None = None,
    ) -> Completion:
        self.totals.waited_seconds += self.budget.wait_for(self.expected_tokens)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_completion_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": dict(json_schema)},
            }
        try:
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except groq.BadRequestError as exc:
            # Groq validates the model's output server-side and rejects a reply
            # that breaks the request: a JSON schema mismatch (sometimes an empty
            # reply), or a call to a tool that wasn't offered (gpt-oss sometimes
            # leaks its internal "commentary" channel as a tool call).
            if "json_validate_failed" in str(exc):
                raise StructuredOutputError(str(exc)) from exc
            if "tool_use_failed" in str(exc):
                raise InvalidToolCallError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except groq.RateLimitError as exc:  # still limited after the SDK's retries
            raise RateLimitedError(str(exc)) from exc
        except groq.APIError as exc:  # timeouts, 5xx after retries
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
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
    """A fake LLMClient for tests: returns prepared completions in order, records requests.

    A reply that is an exception is raised instead, to simulate a failing call.
    """

    replies: list[Completion | Exception]
    requests: list[dict[str, Any]] = field(default_factory=list)
    totals: UsageTotals = field(default_factory=UsageTotals)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        json_schema: Mapping[str, Any] | None = None,
        max_tokens: int = 4096,
        effort: Effort | None = None,
    ) -> Completion:
        self.requests.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "effort": effort,
            }
        )
        if not self.replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        self.totals.add(reply.usage)
        return reply
