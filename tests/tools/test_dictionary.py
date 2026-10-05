from pathlib import Path

import pytest

from cryptic_agent.tools.dictionary import Dictionary, DictionaryNotFoundError


def test_words_are_normalized_and_deduplicated() -> None:
    dictionary = Dictionary(["Treason", "TREASON", "ice-cream", "", "  "])

    assert len(dictionary) == 2
    assert "treason" in dictionary
    assert "ICECREAM" in dictionary


def test_non_strings_are_never_members() -> None:
    assert 7 not in Dictionary(["SEVEN"])


def test_anagrams_share_sorted_letters() -> None:
    dictionary = Dictionary(["senator", "treason", "atoners", "cat"])

    assert dictionary.anagrams("Senator!") == ["ATONERS", "SENATOR", "TREASON"]
    assert dictionary.anagrams("dog") == []


def test_anagrams_returns_a_copy() -> None:
    dictionary = Dictionary(["cat", "act"])

    dictionary.anagrams("cat").clear()

    assert dictionary.anagrams("cat") == ["ACT", "CAT"]


def test_from_ukacd_loads_entries_and_phrase_words(fake_ukacd_zip: Path) -> None:
    dictionary = Dictionary.from_ukacd(fake_ukacd_zip)

    assert "Verdi" in dictionary
    assert "NOSWEAT" in dictionary  # the whole phrase, for a (2,5) answer
    assert "SWEAT" in dictionary  # and each word, for word-by-word checks
    assert "AARONSROD" in dictionary


def test_from_ukacd_never_guesses_lost_accents(fake_ukacd_zip: Path) -> None:
    dictionary = Dictionary.from_ukacd(fake_ukacd_zip)

    assert "PRCIS" not in dictionary
    assert "PRECIS" not in dictionary  # not until another source attests it


def test_missing_word_list_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(DictionaryNotFoundError, match="cryptic-agent ingest"):
        Dictionary.from_ukacd(tmp_path / "nothing-here.zip")
