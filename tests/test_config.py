from pathlib import Path

import pytest

from cryptic_agent import config


def test_data_dirs_default_to_cwd_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DATA_DIR_ENV_VAR, raising=False)

    assert config.data_dir() == Path("data")
    assert config.raw_dir() == Path("data/raw")
    assert config.processed_dir() == Path("data/processed")


def test_data_dir_can_be_overridden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(config.DATA_DIR_ENV_VAR, str(tmp_path))

    assert config.raw_dir() == tmp_path / "raw"


def test_get_api_key_returns_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.API_KEY_ENV_VAR, "sk-test")

    assert config.get_api_key() == "sk-test"


@pytest.mark.parametrize("value", [None, ""])
def test_get_api_key_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(config.API_KEY_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(config.API_KEY_ENV_VAR, value)

    with pytest.raises(config.MissingAPIKeyError, match=config.API_KEY_ENV_VAR):
        config.get_api_key()
