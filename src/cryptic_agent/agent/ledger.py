"""Where a solve's tokens went, call by call.

Every model call in a solve is a "call" step carrying its usage. The ledger lays
them out in order with a running total, splitting what the model wrote into its
hidden reasoning and its visible answer:

    sent       everything in the request: instructions, tool descriptions, and the
               whole conversation so far (so it grows each round)
    reasoning  the model thinking before it answers; usually the largest part
    answer     what it actually said: tool requests or the worksheet
"""

import json
from dataclasses import dataclass

from cryptic_agent.agent.solver import SUBMIT_TOOL, SYSTEM_PROMPT, SolveResult
from cryptic_agent.agent.tools import Toolbox

CHARS_PER_TOKEN = 4  # a rough rule of thumb for English text and JSON


@dataclass(frozen=True)
class LedgerRow:
    step: str  # "tier 2: round 3"
    sent: int
    reasoning: int
    answer: int
    total: int
    running_total: int
    did: str  # what the model did with this call


def ledger(result: SolveResult) -> list[LedgerRow]:
    """One row per model call, in order."""
    rows: list[LedgerRow] = []
    running = 0
    steps = result.steps
    for i, step in enumerate(steps):
        if step.kind != "call" or step.usage is None:
            continue
        usage = step.usage
        running += usage.total_tokens
        # What followed this call, up to the next call: the tools it asked for, or a worksheet.
        after = []
        for later in steps[i + 1 :]:
            if later.kind == "call":
                break
            if later.kind == "tool":
                after.append(later.tool)
            elif later.kind == "worksheet":
                after.append("handed in the worksheet")
        rows.append(
            LedgerRow(
                step=step.text,
                sent=usage.prompt_tokens,
                reasoning=usage.reasoning_tokens,
                answer=usage.completion_tokens - usage.reasoning_tokens,
                total=usage.total_tokens,
                running_total=running,
                did=", ".join(after) or "replied in words",
            )
        )
    return rows


def format_ledger(result: SolveResult) -> str:
    rows = ledger(result)
    if not rows:
        return "No model calls: solved by code alone (0 tokens)."
    header = f"{'step':<34}{'sent':>7}{'reasoning':>11}{'answer':>8}{'total':>8}{'running':>9}  did"
    lines = [header, "-" * len(header)]
    lines += [
        f"{r.step:<34}{r.sent:>7,}{r.reasoning:>11,}{r.answer:>8,}"
        f"{r.total:>8,}{r.running_total:>9,}  {r.did}"
        for r in rows
    ]
    usage = result.usage
    answer = usage.completion_tokens - usage.reasoning_tokens
    lines += [
        "-" * len(header),
        f"{'total':<34}{usage.prompt_tokens:>7,}{usage.reasoning_tokens:>11,}{answer:>8,}"
        f"{usage.total_tokens:>8,}",
    ]
    return "\n".join(lines)


def request_overhead(toolbox: Toolbox) -> dict[str, int]:
    """Roughly how many tokens every tier-2 request carries before any conversation.

    These are re-sent with every round, so they multiply by the number of rounds.
    """
    tools = json.dumps([*toolbox.specs(), SUBMIT_TOOL])
    return {
        "instructions": len(SYSTEM_PROMPT) // CHARS_PER_TOKEN,
        "tool descriptions": len(tools) // CHARS_PER_TOKEN,
    }
