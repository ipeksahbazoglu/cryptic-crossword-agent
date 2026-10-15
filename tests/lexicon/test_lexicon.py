from pathlib import Path

import pytest

from cryptic_agent.lexicon.build import read_curated_abbreviations, repair_damaged
from cryptic_agent.lexicon.store import Evidence, Lexicon, LexiconNotFoundError, url_variants
from cryptic_agent.tools.dictionary import Dictionary

# --- definitions ---------------------------------------------------------------


def test_definition_answers_are_counted_across_past_clues(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("Love god") == [Evidence("EROS", 3)]


def test_definition_lookup_ignores_case_punctuation_and_accents(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("LOVE GOD!") == [Evidence("EROS", 3)]
    assert lexicon.definition_answers("summary") == [Evidence("PRECIS", 1)]  # stored from PRÉCIS


def test_definition_answers_filter_by_length(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("love god", length=5) == []


def test_double_definitions_are_split(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("ruined") == [Evidence("WREN", 1)]
    assert lexicon.definition_answers("a sculpture") == [Evidence("WREN", 1)]


def test_cross_references_are_not_definitions(lexicon: Lexicon) -> None:
    assert lexicon.definition_answers("2") == []


def test_excluded_puzzles_are_invisible(lexicon: Lexicon, guardian_urls: list[str]) -> None:
    # During evaluation the clue under test must not be findable.
    evidence = lexicon.definition_answers("love god", exclude_urls=guardian_urls)

    assert evidence == [Evidence("EROS", 1)]


# --- thesaurus -----------------------------------------------------------------


def test_synonyms_go_both_ways(lexicon: Lexicon) -> None:
    assert lexicon.synonyms("joyful", length=7) == ["FESTIVE"]
    assert lexicon.synonyms("festive") == ["JOYFUL"]


# --- abbreviations ---------------------------------------------------------------


def test_mined_abbreviations_need_enough_support(lexicon: Lexicon) -> None:
    sailor = lexicon.abbreviations("sailor")

    assert Evidence("AB", 3, curated=True) in sailor  # mined 3 times and curated
    assert "SALT" not in [e.value for e in sailor]  # mined once
    assert "SALT" in [e.value for e in lexicon.abbreviations("sailor", min_support=1)]


def test_noise_seen_once_is_filtered(lexicon: Lexicon) -> None:
    assert "GR" not in [e.value for e in lexicon.abbreviations("a")]


def test_curated_abbreviations_fill_gaps_in_the_data(lexicon: Lexicon) -> None:
    # "one" -> I is a crossword staple the mined charades miss.
    assert Evidence("I", 0, curated=True) in lexicon.abbreviations("one")


def test_curated_list_parses() -> None:
    pairs = set(read_curated_abbreviations())

    assert ("sailor", "AB") in pairs
    assert ("the french", "LE") in pairs
    assert all(phrase and letters.isalpha() for phrase, letters in pairs)


# --- indicators -----------------------------------------------------------------


def test_exclusion_matches_any_spelling_of_the_url(lexicon: Lexicon) -> None:
    # The dataset may store https://www.…/ while our scraper sees https://…
    spellings = ["http://www.fifteensquared.net/guardian-1", "fifteensquared.net/guardian-2/"]

    assert lexicon.definition_answers("love god", exclude_urls=spellings) == [Evidence("EROS", 1)]


def test_url_variants() -> None:
    variants = url_variants("https://fifteensquared.net/2021/04/26/guardian-28429/")

    assert "https://www.fifteensquared.net/2021/04/26/guardian-28429/" in variants
    assert "http://fifteensquared.net/2021/04/26/guardian-28429" in variants
    assert len(variants) == 8


def test_excluded_puzzles_drop_out_of_every_lookup(
    lexicon: Lexicon, guardian_urls: list[str]
) -> None:
    sailor = lexicon.abbreviations("sailor", min_support=1, exclude_urls=guardian_urls)
    assert Evidence("AB", 1, curated=True) in sailor  # 3 mined, 2 excluded; curated stays
    assert lexicon.indicator_types("in", exclude_urls=guardian_urls) == []


def test_indicator_types_use_our_names(lexicon: Lexicon) -> None:
    assert lexicon.indicator_types("in") == [Evidence("container", 1)]  # "insertion" mapped
    assert lexicon.indicator_types("about") == [
        Evidence("container", 1),
        Evidence("reversal", 1),
    ]


def test_several_indicators_in_one_cell_are_split(lexicon: Lexicon) -> None:
    assert lexicon.indicator_types("broken") == [Evidence("anagram", 1)]


# --- dictionary words --------------------------------------------------------------


def test_extra_words_need_two_past_clues(lexicon: Lexicon) -> None:
    words = set(lexicon.extra_words())

    assert "EROS" in words  # answer of 3 clues
    assert "WREN" not in words  # answer of 1 clue


def test_damaged_ukacd_entries_restored_only_when_attested(lexicon: Lexicon) -> None:
    # "pr�cis" in the fake UKACD; "precis" is attested by Moby and past answers.
    assert "PRECIS" in lexicon.extra_words()


def test_repair_skips_ambiguous_and_unattested_entries() -> None:
    repaired = repair_damaged(["caf�", "pr�cis", "x�y"], attested=["cafe", "caff", "précis"])

    assert repaired == {"pr�cis": "precis"}  # cafe/caff ambiguous; xiy unattested


def test_dictionary_load_includes_lexicon_words(lexicon_path: Path, fake_ukacd_zip: Path) -> None:
    lexicon = Lexicon(lexicon_path)
    dictionary = Dictionary.from_ukacd(fake_ukacd_zip, extra_words=lexicon.extra_words())
    lexicon.close()

    assert "EROS" in dictionary  # from past answers
    assert "VERDI" in dictionary  # from UKACD


def test_missing_lexicon_says_how_to_build_it(tmp_path: Path) -> None:
    with pytest.raises(LexiconNotFoundError, match="cryptic-agent ingest"):
        Lexicon(tmp_path / "missing.sqlite")
