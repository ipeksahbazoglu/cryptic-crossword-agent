"""The MCP server, tested through a real MCP client: in memory, and once over stdio.

Uses the miniature lexicon from conftest.py ("Love god" -> EROS, "sparkling" an
anagram indicator), so nothing here needs the network or a model.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import StdioServerParameters
from mcp.client import Client

from cryptic_agent.agent.tools import Toolbox
from cryptic_agent.lexicon.sources import UKACD
from cryptic_agent.lexicon.store import Lexicon
from cryptic_agent.mcp_server import build_server
from cryptic_agent.tools.dictionary import Dictionary

pytestmark = pytest.mark.anyio

DICTIONARY = Dictionary(["eros", "ores", "roes", "sore", "rose", "treason", "senator"])
CLUE = {"clue": "Love god's sparkling rose", "enumeration": "4"}


def worksheet(answer: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "definitions": [{"text": "Love god's", "position": "start"}],
        "wordplay": [
            {
                "mechanism": "anagram",
                "indicator": "sparkling",
                "fodder": "rose",
                "produces": answer,
                "explanation": "ROSE rearranged",
            }
        ],
        "link_words": [],
        "confidence": "confirmed",
        "alternatives": [],
    }


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def client(lexicon: Lexicon) -> Client:
    return Client(build_server(DICTIONARY, lexicon))


async def call(client: Client, tool: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    async with client:
        result = await client.call_tool(tool, arguments)
    text = result.content[0].text if result.content else ""  # type: ignore[union-attr]
    return bool(result.is_error), text


# --- what the server offers --------------------------------------------------------


async def test_tools_are_the_agent_toolbox_plus_fast_pass_and_verify(
    client: Client, lexicon: Lexicon
) -> None:
    async with client:
        listed = {t.name for t in (await client.list_tools()).tools}

    # Generated from the same registry as the Groq agent, so the two can't drift.
    assert listed == {*Toolbox(DICTIONARY, lexicon).tools, "fast_pass", "verify_parse"}


async def test_instructions_say_only_verify_parse_confirms(client: Client) -> None:
    async with client:
        assert "verify_parse" in (client.instructions or "")


async def test_solve_prompt_carries_the_method_and_the_clue(client: Client) -> None:
    async with client:
        prompts = await client.list_prompts()
        prompt = await client.get_prompt("solve_cryptic", {**CLUE, "pattern": "E???"})

    assert [p.name for p in prompts.prompts] == ["solve_cryptic"]
    text = prompt.messages[0].content.text  # type: ignore[union-attr]
    assert "Clue: Love god's sparkling rose (4)" in text
    assert "E???" in text and "verify_parse" in text


# --- the tools ------------------------------------------------------------------------


async def test_a_toolbox_tool_runs(client: Client) -> None:
    is_error, text = await call(client, "find_anagrams", {"letters": "senator"})

    assert not is_error and "TREASON" in text


async def test_fast_pass_finds_the_answer_from_both_sides(client: Client) -> None:
    is_error, text = await call(client, "fast_pass", CLUE)

    assert not is_error and "EROS [STRONG]" in text


@pytest.mark.parametrize(("answer", "status"), [("EROS", "confirmed"), ("SORE", "pencilled")])
async def test_verify_parse_gives_the_code_checked_verdict(
    client: Client, answer: str, status: str
) -> None:
    # SORE is also an anagram of "rose", but nothing says "Love god" means SORE.
    is_error, text = await call(client, "verify_parse", {**CLUE, "worksheet": worksheet(answer)})

    assert not is_error and f'"status": "{status}"' in text


# --- untrusted input --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("find_anagrams", {}, "'letters' is a required property"),
        ("find_anagrams", {"letters": 123}, "123 is not of type 'string'"),
        ("verify_parse", {**CLUE, "worksheet": {"answer": "EROS"}}, "is a required property"),
        ("solve_it_for_me", {}, "Unknown tool 'solve_it_for_me'"),
    ],
)
async def test_bad_calls_get_a_clear_error_not_a_crash(
    client: Client, tool: str, arguments: dict[str, Any], message: str
) -> None:
    is_error, text = await call(client, tool, arguments)

    assert is_error and message in text


# --- the real process, as Claude starts it ------------------------------------------------


def data_dir_with_lexicon(root: Path, fake_ukacd_zip: Path, lexicon_path: Path) -> Path:
    """data/lexicon/{lexicon.sqlite, sources/<ukacd zip>}, as `cryptic-agent ingest` leaves it."""
    sources = root / "data" / "lexicon" / "sources"
    sources.mkdir(parents=True)
    shutil.copy(fake_ukacd_zip, sources / UKACD.filename)
    shutil.copy(lexicon_path, sources.parent / "lexicon.sqlite")
    return root / "data"


async def test_the_stdio_server_answers_a_real_client(
    tmp_path: Path, fake_ukacd_zip: Path, lexicon_path: Path
) -> None:
    data = data_dir_with_lexicon(tmp_path / "run", fake_ukacd_zip, lexicon_path)
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cryptic_agent", "mcp"],
        env={**os.environ, "CRYPTIC_AGENT_DATA_DIR": str(data)},
        cwd=str(tmp_path),
    )

    is_error, text = await call(Client(server), "fast_pass", CLUE)

    assert not is_error and "EROS [STRONG]" in text


def test_missing_lexicon_fails_at_startup_with_a_clear_message(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-m", "cryptic_agent", "mcp"],
        env={**os.environ, "CRYPTIC_AGENT_DATA_DIR": str(tmp_path / "empty")},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert run.returncode == 2
    assert "cryptic-agent ingest" in run.stderr
    # stdout carries the protocol. This SDK's client skips stray lines, but stricter
    # clients don't, so nothing may print there; this catches a print during startup.
    assert run.stdout == ""
