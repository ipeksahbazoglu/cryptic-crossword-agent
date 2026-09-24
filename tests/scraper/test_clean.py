import json
from pathlib import Path
from typing import Any

import pytest

from cryptic_agent.scraper.clean import clean_post, html_to_markdown

FIXTURE = Path(__file__).parent.parent / "fixtures" / "wp_posts_quick_cryptic.json"


@pytest.fixture
def raw_posts() -> list[dict[str, Any]]:
    """Two real Quick Cryptic posts from the API, trimmed to the legend and two clues."""
    posts: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return posts


def test_clean_post_maps_api_fields(raw_posts: list[dict[str, Any]]) -> None:
    post = clean_post(raw_posts[0])

    assert post.id == 211874
    assert (
        post.url == "https://fifteensquared.net/2026/09/19/guardian-quick-cryptic-129-by-chandler/"
    )
    assert post.title == "Guardian Quick Cryptic 129 by Chandler"
    assert post.date.year == 2026


def test_real_post_keeps_indicator_and_definition_markup(raw_posts: list[dict[str, Any]]) -> None:
    text = clean_post(raw_posts[0]).content_markdown

    assert (
        "<color=red>After spending money</color>, vigorously bite <u>**small cut of meat**</u> (4)"
        in text
    )
    assert "the indicator is in red" in text  # the blogger's legend survives for extraction


def test_double_definition_keeps_both_definitions(raw_posts: list[dict[str, Any]]) -> None:
    assert "<u>**Outside of**</u> <u>**pub**</u>? (3)" in clean_post(raw_posts[1]).content_markdown


def test_title_entities_are_unescaped(raw_posts: list[dict[str, Any]]) -> None:
    raw = {**raw_posts[0], "title": {"rendered": "Guardian 29,000 &#8211; Paul"}}

    assert clean_post(raw).title == "Guardian 29,000 – Paul"


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ('<span style="color: red">wild</span>', "<color=red>wild</color>"),
        ('<span style="COLOR:#FF0000;">wild</span>', "<color=#ff0000>wild</color>"),
        ('<span style="text-decoration: underline">crime</span>', "<u>crime</u>"),
        ("<u>crime</u>", "<u>crime</u>"),
        ("<strong>crime</strong>", "**crime**"),
        ("<em>anagram</em>", "*anagram*"),
        ('<span style="font-size: 12px">plain</span>', "plain"),
        ('<span style="color: red"> </span>x', "x"),  # no empty tags around whitespace
    ],
)
def test_formatting_is_kept_as_inline_markup(html: str, expected: str) -> None:
    assert html_to_markdown(html) == expected


def test_links_images_and_buttons_are_stripped_to_text() -> None:
    html = '<p><a href="https://x">here</a> <img src="y.png"><button>Expand All</button></p>'

    assert html_to_markdown(html) == "here Expand All"


def test_blank_lines_are_collapsed() -> None:
    # The POC's version of this had a bug (`or True`) that made it a no-op.
    html = "<p>one</p>" + "<p></p>" * 5 + "<p>two</p>"

    assert html_to_markdown(html) == "one\n\ntwo"
