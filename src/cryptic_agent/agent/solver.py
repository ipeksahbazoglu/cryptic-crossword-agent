"""Solving a clue, spending model tokens only when cheaper means fail.

The "tiered" strategy escalates like an expert who doesn't deliberate over easy clues:

    tier 0  code only (0 tokens): the fast pass explains the clue unambiguously and
            code writes the whole worksheet (see assemble.py)
    tier 1  one model call: clue + fast-pass evidence -> worksheet, no tool loop
    tier 2  the agent: reasoning with tools, answering via a submit_answer tool,
            told what tier 1 tried and why it couldn't be confirmed

Every tier ends in the same verify(); a later tier runs only if the earlier one
isn't CONFIRMED. The "agent" strategy skips straight to tier 2 (as before tiers
existed), for comparison runs.

Every step is recorded in `SolveResult.steps`, so a solve can be replayed.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from cryptic_agent.agent.assemble import solve_by_code
from cryptic_agent.agent.fastpass import FastPass
from cryptic_agent.agent.fastpass import fast_pass as run_fast_pass
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
    RateLimitedError,
    StructuredOutputError,
    Usage,
)

logger = logging.getLogger(__name__)

Strategy = Literal["tiered", "agent"]
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

METHOD = """\
You solve UK cryptic crossword clues the way an expert does.

Every clue = a DEFINITION (a plain synonym at the very start or very end) plus
WORDPLAY that builds the same answer another way. Every word has a role:
definition, indicator (signals the mechanism, adds no letters), fodder (what the
mechanism works on), or a short link word ("for", "in", "to", "and", "is").
Mechanisms: anagram, charade, hidden_word, reversal, homophone, container,
deletion, acrostic, alternation, double_definition, cryptic_definition."""

SYSTEM_PROMPT = (
    METHOD
    + """

Method:
1. Note the enumeration and any known letters. Ignore the surface story.
2. Try the definition at BOTH ends: definition_answers and synonyms, with the length.
3. Look for indicators (indicator_types) and their fodder. Quick wins first:
   hidden words, anagrams of words with exactly the right letter count, first letters.
4. Make the two sides meet on one answer. Verify the mechanics with tools; never
   assume an anagram, hidden word or letter selection without checking it.
5. If stuck: swap which end is the definition, read words as other parts of
   speech, split words, try abbreviations.

Be economical: request several tools at once. When you have your answer, call
submit_answer with the full worksheet."""
)

WORKSHEET_GUIDE = """\
Use the clue's exact words for definitions, indicators and fodder. List every
remaining clue word as a link word only if it truly has no other role. confidence:
"confirmed" only if both the definition and the wordplay independently give the
answer; "pencilled" if one side is uncertain; "unsure" otherwise. Give runner-up
answers in alternatives."""

WORKSHEET_PROMPT = "Now fill in the worksheet for your final answer. " + WORKSHEET_GUIDE

QUICK_PROMPT = (
    METHOD
    + """

Solve the clue below in one go and fill in the worksheet. The fast pass lists
candidates found mechanically: use them if they genuinely explain the clue, but
check every word's role yourself. """
    + WORKSHEET_GUIDE
)

RETRY_WORKSHEET_PROMPT = """\
Your worksheet could not be read. Reply with ONLY the JSON worksheet, keeping any
thinking very short."""

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Hand in your final answer as a full worksheet. " + WORKSHEET_GUIDE,
        "parameters": WORKSHEET_SCHEMA,
    },
}


@dataclass(frozen=True)
class Step:
    """One recorded event in a solve."""

    kind: str  # "fastpass" | "tier" | "call" | "thought" | "tool" | "worksheet"
    text: str = ""
    tool: str = ""
    arguments: dict[str, object] = field(default_factory=dict)
    result: str = ""
    usage: Usage | None = None  # set on "call" steps: what that model call cost


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
    tier: int | None = None  # 0 code only, 1 one call, 2 the agent

    @property
    def answer(self) -> str | None:
        return self.verdict.answer if self.verdict else None


def _user_prompt(clue: str, enumeration: str, pattern: str | None) -> str:
    prompt = f"Clue: {clue} ({enumeration})"
    if pattern:
        prompt += f"\nKnown letters from crossing answers (? = unknown): {pattern}"
    return prompt


@dataclass
class _Tally:
    """Tokens used across all tiers of one solve."""

    prompt: int = 0
    completion: int = 0
    reasoning: int = 0

    def add(self, reply: Completion) -> None:
        self.prompt += reply.usage.prompt_tokens
        self.completion += reply.usage.completion_tokens
        self.reasoning += reply.usage.reasoning_tokens

    def usage(self) -> Usage:
        return Usage(self.prompt, self.completion, self.reasoning)


@dataclass
class Solver:
    llm: LLMClient
    toolbox: Toolbox
    max_rounds: int = 5
    on_step: Callable[[Step], None] | None = None  # e.g. print steps live
    fast_pass: bool = True  # give the model mechanically found candidates up front
    strategy: Strategy = "tiered"

    def _record(self, steps: list[Step], step: Step) -> None:
        steps.append(step)
        if self.on_step:
            self.on_step(step)

    def solve(self, clue: str, enumeration: str, *, pattern: str | None = None) -> SolveResult:
        """Solve one clue. Model and API problems become `error`, except running out of
        quota (RateLimitedError), which is raised so the caller can stop and resume."""
        steps: list[Step] = []
        tally = _Tally()
        try:
            return self._solve(clue, enumeration, pattern, steps, tally)
        except RateLimitedError:
            raise  # out of quota: not this clue's fault, so the caller must stop
        except LLMError as exc:
            logger.warning("solve failed: %s", exc)
            return SolveResult(
                clue, enumeration, pattern, steps, None, None, tally.usage(), error=str(exc)
            )

    def _verified(
        self,
        worksheet: Worksheet,
        clue: str,
        enumeration: str,
        pattern: str | None,
        steps: list[Step],
    ) -> Verdict:
        self._record(steps, Step("worksheet", text=json.dumps(worksheet.model_dump(), indent=1)))
        return verify(
            worksheet,
            clue,
            enumeration,
            self.toolbox.dictionary,
            pattern=pattern,
            lexicon=self.toolbox.lexicon,
            exclude_urls=self.toolbox.exclude_urls,
        )

    def _solve(
        self,
        clue: str,
        enumeration: str,
        pattern: str | None,
        steps: list[Step],
        tally: _Tally,
    ) -> SolveResult:
        def result(sheet: Worksheet | None, verdict: Verdict | None, tier: int) -> SolveResult:
            return SolveResult(
                clue, enumeration, pattern, steps, sheet, verdict, tally.usage(), tier=tier
            )

        found: FastPass | None = None
        if self.fast_pass or self.strategy == "tiered":
            found = run_fast_pass(
                clue,
                enumeration,
                self.toolbox.dictionary,
                self.toolbox.lexicon,
                pattern=pattern,
                exclude_urls=self.toolbox.exclude_urls,
            )
            self._record(steps, Step("fastpass", text=found.summary()))

        earlier = ""
        if self.strategy == "tiered" and found is not None:
            self._record(steps, Step("tier", text="tier 0: code only"))
            by_code = solve_by_code(
                found,
                self.toolbox.dictionary,
                self.toolbox.lexicon,
                pattern=pattern,
                exclude_urls=self.toolbox.exclude_urls,
            )
            if by_code is not None:
                sheet, _ = by_code
                return result(sheet, self._verified(sheet, clue, enumeration, pattern, steps), 0)

            self._record(steps, Step("tier", text="tier 1: one model call"))
            sheet1 = self._quick_worksheet(clue, enumeration, pattern, found, steps, tally)
            if sheet1 is not None:
                verdict1 = self._verified(sheet1, clue, enumeration, pattern, steps)
                if verdict1.status == "confirmed":
                    return result(sheet1, verdict1, 1)
                problems = "; ".join(
                    f"{c.name}: {c.detail}" for c in verdict1.checks if not c.passed
                )
                earlier = (
                    f"\n\nA quick first attempt answered {verdict1.answer}, but it could not be "
                    f"confirmed ({problems}). Reconsider it, or find something better."
                )
            self._record(steps, Step("tier", text="tier 2: the agent with tools"))

        sheet2 = self._agent_worksheet(clue, enumeration, pattern, found, earlier, steps, tally)
        if sheet2 is None:
            return SolveResult(
                clue,
                enumeration,
                pattern,
                steps,
                None,
                None,
                tally.usage(),
                error="the model could not produce a valid worksheet",
                tier=2,
            )
        return result(sheet2, self._verified(sheet2, clue, enumeration, pattern, steps), 2)

    def _count(self, reply: Completion, label: str, steps: list[Step], tally: _Tally) -> None:
        """Record one model call: its cost goes in the tally and in the step log."""
        tally.add(reply)
        self._record(steps, Step("call", text=label, usage=reply.usage))

    def _rejected(self, label: str, steps: list[Step]) -> None:
        """Log a call the provider rejected, so the ledger shows it happened.

        The error reply carries no token count, so its cost is recorded as unknown (0).
        """
        self._record(steps, Step("call", text=f"{label}, rejected", usage=Usage()))

    def _request_worksheet(
        self, messages: list[Message], label: str, steps: list[Step], tally: _Tally
    ) -> Worksheet | None:
        """Ask for a strict-schema worksheet, with retries (see WORKSHEET_ATTEMPTS)."""
        for attempt, (effort, max_tokens) in enumerate(WORKSHEET_ATTEMPTS, start=1):
            try:
                reply = self.llm.complete(
                    messages, json_schema=WORKSHEET_SCHEMA, max_tokens=max_tokens, effort=effort
                )
            except (StructuredOutputError, InvalidToolCallError) as exc:
                logger.warning("worksheet attempt %d (%s effort) failed: %s", attempt, effort, exc)
                self._rejected(f"{label} ({effort} effort)", steps)
                if attempt == 1:
                    messages.append({"role": "user", "content": RETRY_WORKSHEET_PROMPT})
                continue
            self._count(reply, f"{label} ({effort} effort)", steps, tally)
            try:
                return Worksheet.model_validate_json(reply.content)
            except ValidationError as exc:
                logger.warning("worksheet did not validate: %s", exc)
                return None
        return None

    def _quick_worksheet(
        self,
        clue: str,
        enumeration: str,
        pattern: str | None,
        found: FastPass,
        steps: list[Step],
        tally: _Tally,
    ) -> Worksheet | None:
        """Tier 1: one call, no tools."""
        messages: list[Message] = [
            {"role": "system", "content": QUICK_PROMPT},
            {
                "role": "user",
                "content": _user_prompt(clue, enumeration, pattern) + "\n\n" + found.summary(),
            },
        ]
        return self._request_worksheet(messages, "tier 1: worksheet", steps, tally)

    def _agent_worksheet(
        self,
        clue: str,
        enumeration: str,
        pattern: str | None,
        found: FastPass | None,
        earlier: str,
        steps: list[Step],
        tally: _Tally,
    ) -> Worksheet | None:
        """Tier 2: the tool-use loop, answering through submit_answer."""
        user_prompt = _user_prompt(clue, enumeration, pattern)
        if found is not None and self.fast_pass:
            user_prompt += "\n\n" + found.summary()
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt + earlier},
        ]
        tools = [*self.toolbox.specs(), SUBMIT_TOOL]

        for round_number in range(1, self.max_rounds + 1):
            try:
                reply = self.llm.complete(messages, tools=tools)
            except InvalidToolCallError as exc:
                note = f"That tool call was rejected ({exc}). Use only the tools provided."
                self._rejected(f"tier 2: round {round_number}", steps)
                self._record(steps, Step("thought", text=f"[invalid tool call] {exc}"))
                messages.append({"role": "user", "content": note})
                continue
            self._count(reply, f"tier 2: round {round_number}", steps, tally)
            if reply.reasoning or reply.content:
                self._record(
                    steps, Step("thought", text=(reply.reasoning or reply.content).strip())
                )
            messages.append(reply.as_message())
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                if call.name == "submit_answer":
                    try:
                        return Worksheet.model_validate(call.arguments)
                    except ValidationError as exc:
                        output = json.dumps(
                            {"error": f"worksheet invalid, fix and resubmit: {exc}"}
                        )
                else:
                    output = self.toolbox.run(call.name, call.arguments)[:MAX_RESULT_CHARS]
                self._record(
                    steps, Step("tool", tool=call.name, arguments=call.arguments, result=output)
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        else:
            # Out of rounds while still calling tools: a tool message must not be
            # the last word before the worksheet request, so add a nudge.
            messages.append({"role": "user", "content": "Stop using tools now."})

        # The model answered in words instead of submitting: ask for the worksheet.
        messages.append({"role": "user", "content": WORKSHEET_PROMPT})
        return self._request_worksheet(messages, "tier 2: worksheet", steps, tally)
