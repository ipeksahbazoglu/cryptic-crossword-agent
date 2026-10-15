"""The solver's tools for Claude (or any MCP client), over the Model Context Protocol.

With MCP, the client's model does the reasoning and this server supplies what it
can't: a crossword dictionary, 660,000 past clues, mechanical wordplay checks and
the verdict. Everything runs locally; only tool results reach the client.

    fast_pass       candidates code finds alone (the same pre-pass the Groq agent gets)
    verify_parse    checks a full parse; the only way an answer becomes "confirmed"
    9 more          the agent's own tools, taken from the Toolbox registry, so this
                    server and the Groq agent can never offer different tools

Arguments come from the model, so every call is validated against its tool's
schema here: the SDK passes them through unchecked.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from cryptic_agent.agent.fastpass import fast_pass
from cryptic_agent.agent.solver import METHOD, clue_prompt
from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.agent.verify import verify
from cryptic_agent.agent.worksheet import WORKSHEET_SCHEMA, Worksheet
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.tools.dictionary import Dictionary

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Tools for solving UK cryptic crossword clues. For a clue: call fast_pass first,
use the other tools to test ideas, then call verify_parse with your full parse.
Never present an answer as confirmed unless verify_parse returned "confirmed";
otherwise say it is pencilled in and why."""

CLUE_ARGS: dict[str, Any] = {
    "clue": {"type": "string", "description": "The clue, without the enumeration."},
    "enumeration": {"type": "string", "description": "Answer word lengths, e.g. '7' or '3,4'."},
    "pattern": {
        "type": "string",
        "description": "Optional letters from crossing answers, '?' for unknown, e.g. 'T?E?S?N'.",
    },
}


@dataclass(frozen=True)
class _Tool:
    spec: types.Tool
    run: Callable[[dict[str, Any]], str]


def _tool(name: str, description: str, schema: dict[str, Any], run: Callable[..., str]) -> _Tool:
    return _Tool(types.Tool(name=name, description=description, input_schema=schema), run)


def build_server(dictionary: Dictionary, lexicon: Lexicon) -> Server[Any]:
    """An MCP server over this dictionary and lexicon (not yet connected to anything)."""
    toolbox = Toolbox(dictionary, lexicon)

    def run_fast_pass(args: dict[str, Any]) -> str:
        found = fast_pass(
            args["clue"], args["enumeration"], dictionary, lexicon, pattern=args.get("pattern")
        )
        return found.summary()

    def run_verify(args: dict[str, Any]) -> str:
        worksheet = Worksheet.model_validate(args["worksheet"])
        verdict = verify(
            worksheet,
            args["clue"],
            args["enumeration"],
            dictionary,
            pattern=args.get("pattern"),
            lexicon=lexicon,
        )
        return verdict.model_dump_json(indent=1)

    tools = {
        t.spec.name: t
        for t in [
            _tool(
                "fast_pass",
                "Start here. Candidates found by code alone: definitions at either end of "
                "the clue, and anagrams, hidden words and letter selections of its words. "
                "STRONG means a definition and separate wordplay agree.",
                {"type": "object", "properties": CLUE_ARGS, "required": ["clue", "enumeration"]},
                run_fast_pass,
            ),
            _tool(
                "verify_parse",
                "Finish here. Code checks your full parse: length, real word, definition at "
                "an end, wordplay mechanics, every word used once. Returns confirmed, "
                "pencilled or unsure, with each check. Only this can confirm an answer.",
                {
                    "type": "object",
                    "properties": {**CLUE_ARGS, "worksheet": WORKSHEET_SCHEMA},
                    "required": ["clue", "enumeration", "worksheet"],
                },
                run_verify,
            ),
            *(
                _tool(
                    tool.name,
                    tool.description,
                    tool.parameters,
                    lambda args, name=tool.name: toolbox.run(name, args),
                )
                for tool in toolbox.tools.values()
            ),
        ]
    }
    validators = {name: Draft202012Validator(t.spec.input_schema) for name, t in tools.items()}

    def error(message: str) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)], is_error=True
        )

    async def list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[t.spec for t in tools.values()])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        tool = tools.get(params.name)
        if tool is None:
            return error(f"Unknown tool {params.name!r}. Available: {', '.join(sorted(tools))}.")
        arguments = params.arguments or {}
        problems = [e.message for e in validators[params.name].iter_errors(arguments)]
        if problems:
            return error("Invalid arguments: " + "; ".join(problems))
        try:
            text = tool.run(arguments)
        except ValidationError as exc:  # a worksheet the schema allows but the model rejects
            return error(f"Invalid worksheet: {exc}")
        except Exception:
            logger.exception("tool %s failed", params.name)  # stderr: stdout is the protocol
            return error(f"{params.name} failed unexpectedly; see the server log.")
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

    async def list_prompts(ctx: Any, params: Any) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name="solve_cryptic",
                    description="Solve a cryptic clue step by step with the tools.",
                    arguments=[
                        types.PromptArgument(name="clue", required=True),
                        types.PromptArgument(name="enumeration", required=True),
                        types.PromptArgument(name="pattern", required=False),
                    ],
                )
            ]
        )

    async def get_prompt(ctx: Any, params: types.GetPromptRequestParams) -> types.GetPromptResult:
        args = params.arguments or {}
        if params.name != "solve_cryptic" or not args.get("clue") or not args.get("enumeration"):
            raise ValueError("solve_cryptic needs a clue and an enumeration")
        text = (
            f"{METHOD}\n\n{INSTRUCTIONS}\n\n"
            f"{clue_prompt(args['clue'], args['enumeration'], args.get('pattern') or None)}"
        )
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))
            ]
        )

    return Server(
        "cryptic-crossword",
        instructions=INSTRUCTIONS,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_prompts=list_prompts,
        on_get_prompt=get_prompt,
    )


async def serve() -> None:
    """Run over stdio until the client disconnects. Loads the data first, so a
    missing lexicon fails at startup with a clear message, not on the first clue."""
    server = build_server(Dictionary.load(), Lexicon())
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
