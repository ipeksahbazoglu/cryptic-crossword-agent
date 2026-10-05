"""Integration tests against the real built lexicon.

Skipped unless `cryptic-agent ingest` has built data/lexicon/lexicon.sqlite. CI
skips them on purpose: building needs a 187 MB download.
"""

from collections.abc import Iterator

import pytest

from cryptic_agent.lexicon.store import Lexicon, LexiconNotFoundError


@pytest.fixture(scope="module")
def lexicon() -> Iterator[Lexicon]:
    try:
        lx = Lexicon()
    except LexiconNotFoundError:
        pytest.skip("lexicon not built; run: uv run cryptic-agent ingest")
    yield lx
    lx.close()


def test_crossword_knowledge_a_thesaurus_lacks(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("Love god", length=4)[0].value == "EROS"
    assert "ATTLEE" in [e.value for e in lexicon.definition_answers("Prime Minister", length=6)]


def test_standard_abbreviations(lexicon: Lexicon) -> None:
    assert {"AB", "TAR"} <= {e.value for e in lexicon.abbreviations("sailor")}
    assert "I" in {e.value for e in lexicon.abbreviations("one")}


def test_indicator_frequencies(lexicon: Lexicon) -> None:
    assert lexicon.indicator_types("cook")[0].value == "anagram"
    assert lexicon.indicator_types("back")[0].value == "reversal"


def test_words_newer_than_ukacd_are_known(lexicon: Lexicon) -> None:
    assert lexicon.is_known_answer("SELFIE")
    assert lexicon.is_known_answer("PRECIS")
