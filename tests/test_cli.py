import argparse
from pathlib import Path

import pytest

from cryptic_agent import cli, config


@pytest.fixture(autouse=True)
def _isolate_from_real_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # main() loads .env from the working directory; run from an empty temp dir
    # so a developer's real .env never leaks into tests.
    monkeypatch.chdir(tmp_path)


def test_help_lists_all_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("scrape", "extract", "solve", "eval"):
        assert command in out


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["scrape", "--category", "guardian"],
        ["extract"],
        ["solve", "--clue", "Senator arranged crime", "--enumeration", "7"],
        ["eval"],
    ],
)
def test_stub_commands_parse_and_report_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 1
    assert "not implemented" in capsys.readouterr().err


def test_parser_applies_types_and_defaults() -> None:
    args = cli.build_parser().parse_args(["eval", "--dataset", "clues.jsonl", "--n", "10"])

    assert args.dataset == Path("clues.jsonl")
    assert args.n == 10
    assert args.seed == 42


def test_missing_api_key_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def needs_key(args: argparse.Namespace) -> int:
        config.get_api_key()
        return 0

    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    # Stand in for a real command that needs the key.
    monkeypatch.setattr(cli, "_not_implemented", needs_key)

    assert cli.main(["eval"]) == 2
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err
