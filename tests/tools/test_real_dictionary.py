"""Integration tests against the real NLTK word list.

Skipped when the corpus is not downloaded; CI downloads it so these always run there.
"""

import pytest

from cryptic_agent.tools.dictionary import Dictionary, DictionaryNotFoundError
from cryptic_agent.tools.wordplay import check_answer, check_hidden_word, find_anagrams


@pytest.fixture(scope="module")
def nltk_dictionary() -> Dictionary:
    # scope="module": build the 234k-word index once for this file, not per test.
    try:
        return Dictionary.from_nltk()
    except DictionaryNotFoundError:
        pytest.skip("NLTK 'words' corpus not installed")


def test_senator_anagrams_to_treason(nltk_dictionary: Dictionary) -> None:
    result = find_anagrams(nltk_dictionary, "SENATOR", length=7)

    assert "TREASON" in result.matches
    assert "SENATOR" not in result.matches


def test_hidden_word_in_real_clue_text(nltk_dictionary: Dictionary) -> None:
    result = check_hidden_word(nltk_dictionary, "Some aroma nce", 7)

    assert "ROMANCE" in [m.word for m in result.matches]


def test_multi_word_answer_checked_word_by_word(nltk_dictionary: Dictionary) -> None:
    assert check_answer(nltk_dictionary, "ICE CREAM", "3,5").valid
