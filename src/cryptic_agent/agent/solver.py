"""The solving loop: the model reasons and calls tools, then hands in a worksheet.

    clue -> [model asks for tools -> we run them -> results back] x up to N rounds
         -> final call: fill in the worksheet (strict JSON schema)
         -> code verifies the worksheet -> verdict: confirmed / pencilled / unsure

Every step is recorded in `SolveResult.steps`, so a solve can be replayed and
inspected: which tools were called, with what, and what came back.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.agent.verify import Verdict, verify
from cryptic_agent.agent.worksheet import WORKSHEET_SCHEMA, Worksheet
from cryptic_agent.llm.client import (
    Completion,
    Effort,
    InvalidToolCallError,
    LLMClient,
    LLMError,
    Message,
    StructuredOutputError,
    Usage,
)

logger = logging.getLogger(__name__)

MAX_RESULT_CHARS = 1200  # cap what one tool result adds to the conversation

# Worksheet attempts, measured live: at medium effort the model writes good
# worksheets but occasionally runs out of room and returns nothing; at low effort
# it always returns JSON but re-solves sloppily (wrong answers, which verification
# then refuses to confirm). So: medium with room to spare, retry, low as a last resort.
WORKSHEET_ATTEMPTS: tuple[tuple[Effort, int], ...] = (
    ("medium", 6000),
    ("medium", 6000),
    ("low", 3000),
)

SYSTEM_PROMPT = """\
You solve UK cryptic crossword clues the way an expert does.

Every clue = a DEFINITION (a plain synonym at the very start or very end) plus
WORDPLAY that builds the same answer another way. Every word has a role:
definition, indicator (signals the mechanism, adds no letters), fodder (what the
mechanism works on), or a short link word ("for", "in", "to", "and", "is").
Mechanisms: anagram, charade, hidden_word, reversal, homophone, container,
deletion, acrostic, alternation, double_definition, cryptic_definition.

Method:
1. Note the enumeration and any known letters. Ignore the surface story.
2. Try the definition at BOTH ends: definition_answers and synonyms, with the length.
3. Look for indicators (indicator_types) and their fodder. Quick wins first:
   hidden words, anagrams of words with exactly the right letter count, first letters.
4. Make the two sides meet on one answer. Verify the mechanics with tools; never
   assume an anagram, hidden word or letter selection without checking it.
5. If stuck: swap which end is the definition, read words as other parts of
   speech, split words, try abbreviations.
6. Call check_answer on your final candidate.

Be economical: request several tools at once when you can. When you have your
answer (or are stuck), reply briefly in words without calling tools."""

WORKSHEET_PROMPT = """\
Now fill in the worksheet for your final answer. Use the clue's exact words for
definitions, indicators and fodder. List every remaining clue word as a link word
only if it truly has no other role. confidence: "confirmed" only if both the
definition and the wordplay independently give the answer; "pencilled" if one
side is uncertain; "unsure" otherwise. Give runner-up answers in alternatives."""


RETRY_WORKSHEET_PROMPT = """\
Your worksheet could not be read. Reply with ONLY the JSON worksheet, keeping any
thinking very short."""


@dataclass(frozen=True)
class Step:
    """One recorded event in a solve."""

    kind: str  # "thought" | "tool" | "worksheet"
    text: str = ""
    tool: str = ""
    arguments: dict[str, object] = field(default_factory=dict)
    result: str = ""


@dataclass
class SolveResult:
    clue: str
    enumeration: str
    pattern: str | None
    steps: list[Step]
    worksheet: Worksheet | None
    verdict: Verdict | None
    usage: Usage
    error: str = ""

    @property
    def answer(self) -> str | None:
        return self.verdict.answer if self.verdict else None


def _user_prompt(clue: str, enumeration: str, pattern: str | None) -> str:
    prompt = f"Clue: {clue} ({enumeration})"
    if pattern:
        prompt += f"\nKnown letters from crossing answers (? = unknown): {pattern}"
    return prompt


@dataclass
class Solver:
    llm: LLMClient
    toolbox: Toolbox
    max_rounds: int = 8
    on_step: Callable[[Step], None] | None = None  # e.g. print steps live

    def _record(self, steps: list[Step], step: Step) -> None:
        steps.append(step)
        if self.on_step:
            self.on_step(step)

    def solve(self, clue: str, enumeration: str, *, pattern: str | None = None) -> SolveResult:
        """Solve one clue. Never raises for model or API problems: they become `error`."""
        steps: list[Step] = []
        try:
            return self._solve(clue, enumeration, pattern, steps)
        except LLMError as exc:
            logger.warning("solve failed: %s", exc)
            return SolveResult(
                clue, enumeration, pattern, steps, None, None, Usage(), error=str(exc)
            )

    def _solve(
        self, clue: str, enumeration: str, pattern: str | None, steps: list[Step]
    ) -> SolveResult:
        prompt = 0
        completion = 0
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(clue, enumeration, pattern)},
        ]

        for _ in range(self.max_rounds):
            try:
                reply = self.llm.complete(messages, tools=self.toolbox.specs())
            except InvalidToolCallError as exc:
                note = f"That tool call was rejected ({exc}). Use only the tools provided."
                self._record(steps, Step("thought", text=f"[invalid tool call] {exc}"))
                messages.append({"role": "user", "content": note})
                continue
            prompt += reply.usage.prompt_tokens
            completion += reply.usage.completion_tokens
            if reply.reasoning or reply.content:
                self._record(
                    steps, Step("thought", text=(reply.reasoning or reply.content).strip())
                )
            messages.append(reply.as_message())
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                result = self.toolbox.run(call.name, call.arguments)[:MAX_RESULT_CHARS]
                self._record(
                    steps, Step("tool", tool=call.name, arguments=call.arguments, result=result)
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        else:
            # Out of rounds while still calling tools: a tool message must not be
            # the last word before the worksheet request, so add a nudge.
            messages.append({"role": "user", "content": "Stop using tools now."})

        messages.append({"role": "user", "content": WORKSHEET_PROMPT})
        final: Completion | None = None
        for attempt, (effort, max_tokens) in enumerate(WORKSHEET_ATTEMPTS, start=1):
            try:
                final = self.llm.complete(
                    messages, json_schema=WORKSHEET_SCHEMA, max_tokens=max_tokens, effort=effort
                )
                break
            except (StructuredOutputError, InvalidToolCallError) as exc:
                logger.warning("worksheet attempt %d (%s effort) failed: %s", attempt, effort, exc)
                if attempt == 1:
                    messages.append({"role": "user", "content": RETRY_WORKSHEET_PROMPT})
        if final is None:
            usage = Usage(prompt, completion)
            error = "the model could not produce a valid worksheet"
            return SolveResult(clue, enumeration, pattern, steps, None, None, usage, error=error)
        prompt += final.usage.prompt_tokens
        completion += final.usage.completion_tokens
        usage = Usage(prompt, completion)

        try:
            worksheet = Worksheet.model_validate_json(final.content)
        except ValidationError as exc:
            logger.warning("worksheet did not validate: %s", exc)
            return SolveResult(clue, enumeration, pattern, steps, None, None, usage, error=str(exc))
        self._record(steps, Step("worksheet", text=json.dumps(worksheet.model_dump(), indent=1)))

        verdict = verify(worksheet, clue, enumeration, self.toolbox.dictionary, pattern=pattern)
        return SolveResult(clue, enumeration, pattern, steps, worksheet, verdict, usage)
