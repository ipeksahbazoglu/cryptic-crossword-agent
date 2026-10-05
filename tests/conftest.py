"""Fixtures shared by several test files (pytest discovers this file automatically)."""

import zipfile
from pathlib import Path

import pytest

from cryptic_agent.lexicon.wordlist import UKACD_MEMBER

FAKE_UKACD = "\n".join(
    [
        "Copyright (c) 2009 J Ross Beresford",
        "All rights reserved.",
        "-" * 68,
        "a",
        "Aaron's rod",
        "eels",
        "no sweat",
        "pr�cis",  # an entry whose accent was lost upstream
        "",
        "senator",
        "treason",
        "Verdi",
    ]
)


@pytest.fixture
def fake_ukacd_zip(tmp_path: Path) -> Path:
    """A miniature copy of the UKACD archive, laid out like the real one."""
    path = tmp_path / "ccxxv-0.1.4.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(UKACD_MEMBER, FAKE_UKACD)
    return path
