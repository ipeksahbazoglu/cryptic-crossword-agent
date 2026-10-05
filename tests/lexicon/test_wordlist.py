from pathlib import Path

import pytest

from cryptic_agent.lexicon.wordlist import damaged_pattern, read_ukacd


def test_read_ukacd_splits_licence_entries_and_damage(fake_ukacd_zip: Path) -> None:
    ukacd = read_ukacd(fake_ukacd_zip)

    assert ukacd.licence.startswith("Copyright (c) 2009 J Ross Beresford")
    assert ukacd.entries == ["a", "Aaron's rod", "eels", "no sweat", "senator", "treason", "Verdi"]
    assert ukacd.damaged == ["pr�cis"]


@pytest.mark.parametrize(
    ("damaged", "word", "fits"),
    [
        ("pr�cis", "precis", True),
        ("pr�cis", "precise", False),  # one lost letter is exactly one letter
        ("� bient�t", "a bientot", True),
        ("caf�", "cafe", True),
        ("caf�", "caff", True),  # the pattern alone can't tell: another source decides
    ],
)
def test_damaged_pattern(damaged: str, word: str, fits: bool) -> None:
    assert bool(damaged_pattern(damaged).fullmatch(word)) is fits
