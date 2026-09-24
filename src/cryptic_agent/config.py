"""Project-wide settings: the one place to change the model, data paths or API key source.

Nothing here does work at import time. Values that depend on the environment are
read through functions, so importing this module never fails and tests can
control the environment with monkeypatch.
"""

import os
from pathlib import Path

MODEL = "claude-sonnet-5"

DATA_DIR_ENV_VAR = "CRYPTIC_AGENT_DATA_DIR"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when a command needs the Anthropic API key and none is configured."""


def data_dir() -> Path:
    """Root directory for scraped and processed data (default: ./data)."""
    return Path(os.environ.get(DATA_DIR_ENV_VAR, "data"))


def raw_dir() -> Path:
    """Where the scraper writes raw posts, one .jsonl file per category."""
    return data_dir() / "raw"


def processed_dir() -> Path:
    """Where extraction writes structured clue records."""
    return data_dir() / "processed"


def get_api_key() -> str:
    """Return the Anthropic API key, failing with a clear message if it is missing.

    Only commands that call the API should call this, so `scrape` and `--help`
    work without a key.
    """
    key = os.environ.get(API_KEY_ENV_VAR)
    if not key:
        raise MissingAPIKeyError(
            f"{API_KEY_ENV_VAR} is not set. Add it to your environment or to a .env file "
            "(see .env.example)."
        )
    return key
