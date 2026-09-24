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


def test_missing_nltk_corpus_gives_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from nltk.corpus import words

    def missing() -> list[str]:
        raise LookupError("Resource 'words' not found.")

    monkeypatch.setattr(words, "words", missing)

    with pytest.raises(DictionaryNotFoundError, match=r"nltk\.download\('words'\)"):
        Dictionary.from_nltk()
