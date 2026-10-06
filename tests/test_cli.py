import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cryptic_agent import cli, config
from cryptic_agent.lexicon.sources import Source
from cryptic_agent.scraper.client import CategoryNotFoundError, WordPressClient


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
        ["extract"],
    ],
)
def test_stub_commands_parse_and_report_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 1
    assert "not implemented" in capsys.readouterr().err


def test_parser_applies_types_and_defaults() -> None:
    args = cli.build_parser().parse_args(["eval", "--n", "10", "--reveal-every", "2"])

    assert args.n == 10
    assert args.reveal_every == 2
    assert args.seed == 42
    assert args.report_only is False


def test_missing_api_key_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def needs_key(args: argparse.Namespace) -> int:
        config.get_api_key()
        return 0

    monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    # Stand in for a real command that needs the key.
    monkeypatch.setattr(cli, "_not_implemented", needs_key)

    assert cli.main(["extract"]) == 2
    assert f"{config.API_KEY_ENV_VAR} is not set" in capsys.readouterr().err


class _StubWordPress(WordPressClient):
    def __init__(self) -> None:
        pass

    def resolve_category_id(self, slug: str) -> int:
        if slug == "missing":
            raise CategoryNotFoundError("no Fifteensquared category with slug 'missing'")
        return 56

    def iter_posts(
        self, category_id: int, *, max_pages: int, per_page: int = 100
    ) -> Iterator[dict[str, Any]]:
        yield {
            "id": 1,
            "date": "2026-09-19T08:20:11",
            "link": "https://fifteensquared.net/1/",
            "title": {"rendered": "Guardian Quick Cryptic 129"},
            "content": {"rendered": "<p>clue</p>"},
        }


def test_scrape_writes_to_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "WordPressClient", _StubWordPress)
    monkeypatch.setenv(config.DATA_DIR_ENV_VAR, str(tmp_path / "data"))

    assert cli.main(["scrape", "--category", "guardian/quick-cryptic"]) == 0

    assert (tmp_path / "data" / "raw" / "guardian_quick-cryptic.jsonl").exists()
    assert "Fetched 1 posts (1 new)" in capsys.readouterr().out


def test_scrape_unknown_category_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "WordPressClient", _StubWordPress)

    assert cli.main(["scrape", "--category", "missing"]) == 2
    assert "no Fifteensquared category" in capsys.readouterr().err


def test_ingest_without_big_sources_skips_the_lexicon_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_fetch(source: Source, dest_dir: Path) -> Path:
        path = dest_dir / source.filename
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr(cli, "fetch", fake_fetch)
    monkeypatch.setenv(config.DATA_DIR_ENV_VAR, str(tmp_path / "data"))

    assert cli.main(["ingest", "--source", "ukacd"]) == 0

    out = capsys.readouterr().out
    assert "ukacd:" in out
    assert "lexicon not built" in out
