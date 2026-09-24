from itertools import permutations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cryptic_agent.tools.dictionary import Dictionary
from cryptic_agent.tools.wordplay import (
    check_answer,
    check_hidden_word,
    find_anagrams,
    reverse_letters,
)

WORDS = Dictionary(
    [
        "senator", "treason", "atoners",  # anagrams of each other
        "roman", "romance", "man", "cerebral", "a",  # hidden-word fodder
        "ice", "cream", "strap", "parts", "cat", "act",
    ]
)  # fmt: skip

letters = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=10)


# --- find_anagrams ----------------------------------------------------------


def test_find_anagrams_excludes_the_fodder_itself() -> None:
    result = find_anagrams(WORDS, "senator")

    assert result.input_letters == "SENATOR"
    assert result.matches == ["ATONERS", "TREASON"]
    assert result.count == 2


def test_find_anagrams_ignores_spaces_and_punctuation() -> None:
    assert find_anagrams(WORDS, "no rates!").matches == ["ATONERS", "SENATOR", "TREASON"]


def test_find_anagrams_length_mismatch_returns_nothing() -> None:
    assert find_anagrams(WORDS, "senator", length=6).matches == []


def test_find_anagrams_caps_matches_but_reports_full_count() -> None:
    many = Dictionary("".join(p) for p in permutations("ABCDE"))

    result = find_anagrams(many, "EDCBA")

    assert len(result.matches) == 50
    assert result.count == 119  # 5! permutations minus the input itself


# Random letters almost never form a word, so shuffle real words instead:
# that way most generated inputs have anagrams and the property is exercised.
shuffled_words = (
    st.sampled_from(["SENATOR", "TREASON", "CAT", "STRAP", "ROMANCE"])
    .flatmap(lambda word: st.permutations(list(word)))
    .map("".join)
)


@given(st.one_of(shuffled_words, letters))
def test_every_anagram_match_uses_exactly_the_input_letters(text: str) -> None:
    result = find_anagrams(WORDS, text)

    assert result.count == len(result.matches)  # WORDS is far below the cap
    for match in result.matches:
        assert match in WORDS
        assert sorted(match) == sorted(text)
        assert match != text


# --- check_hidden_word -------------------------------------------------------


def test_hidden_word_spanning_clue_words() -> None:
    result = check_hidden_word(WORDS, "a Roman cerebral", 7)

    assert result.squashed_text == "AROMANCEREBRAL"
    assert [(m.word, m.spans_words) for m in result.matches] == [("ROMANCE", True)]


def test_whole_clue_words_are_not_hidden_words() -> None:
    result = check_hidden_word(WORDS, "a Roman cerebral", 5)

    assert "ROMAN" not in [m.word for m in result.matches]


def test_single_word_matches_are_kept_but_ranked_after_spanning_ones() -> None:
    # CAT spans chiC+ATom; ACT sits inside the single word trACTor.
    result = check_hidden_word(WORDS, "chic atom tractor", 3)

    assert [(m.word, m.spans_words) for m in result.matches] == [
        ("CAT", True),
        ("ACT", False),
    ]


@pytest.mark.parametrize("length", [0, -1])
def test_hidden_word_rejects_non_positive_length(length: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        check_hidden_word(WORDS, "a Roman cerebral", length)


# --- check_answer --------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "enumeration", "length_ok", "unknown", "valid"),
    [
        ("treason", "7", True, [], True),
        ("ice cream", "3,5", True, [], True),  # checked as ICE + CREAM
        ("ICECREAM", "8", True, ["ICECREAM"], False),  # not one dictionary word
        ("treason", "6", False, [], False),
        ("xyzzy", "5", True, ["XYZZY"], False),
    ],
)
def test_check_answer(
    answer: str, enumeration: str, length_ok: bool, unknown: list[str], valid: bool
) -> None:
    result = check_answer(WORDS, answer, enumeration)

    assert (result.length_ok, result.unknown_words, result.valid) == (length_ok, unknown, valid)


# --- reverse_letters -----------------------------------------------------------


def test_reverse_letters() -> None:
    assert reverse_letters("Strap!").reversed == "PARTS"


@given(letters)
def test_reversing_twice_is_identity(text: str) -> None:
    once = reverse_letters(text).reversed
    assert reverse_letters(once).reversed == text
