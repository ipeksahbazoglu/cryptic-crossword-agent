"""Integration tests against the real UKACD word list.

Skipped until `cryptic-agent ingest` has run; CI runs it so these always run there.
"""

import pytest

from cryptic_agent.tools.dictionary import Dictionary, DictionaryNotFoundError
from cryptic_agent.tools.wordplay import check_answer, check_hidden_word, find_anagrams


@pytest.fixture(scope="module")
def ukacd() -> Dictionary:
    # scope="module": build the 250k-word index once for this file, not per test.
    try:
        return Dictionary.from_ukacd()
    except DictionaryNotFoundError:
        pytest.skip("UKACD not downloaded; run: uv run cryptic-agent ingest")


def test_senator_anagrams_to_treason(ukacd: Dictionary) -> None:
    result = find_anagrams(ukacd, "SENATOR", length=7)

    assert "TREASON" in result.matches
    assert "SENATOR" not in result.matches


def test_hidden_word_in_real_clue_text(ukacd: Dictionary) -> None:
    result = check_hidden_word(ukacd, "Some aroma nce", 7)

    assert "ROMANCE" in [m.word for m in result.matches]


def test_multi_word_answer_checked_word_by_word(ukacd: Dictionary) -> None:
    assert check_answer(ukacd, "ICE CREAM", "3,5").valid


@pytest.mark.parametrize("word", ["VERDI", "SERBIA", "OFFENCE", "EELS", "PERESTROIKA", "NOSWEAT"])
def test_words_nltk_was_missing(ukacd: Dictionary, word: str) -> None:
    assert word in ukacd
