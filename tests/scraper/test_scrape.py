import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from cryptic_agent.jsonl import read_jsonl
from cryptic_agent.models import RawPost
from cryptic_agent.scraper.client import WordPressClient
from cryptic_agent.scraper.scrape import output_path, scrape_category

FIXTURE = Path(__file__).parent.parent / "fixtures" / "wp_posts_quick_cryptic.json"


class FakeWordPress(WordPressClient):
    """Serves fixed posts without any HTTP."""

    def __init__(self, posts: list[dict[str, Any]]) -> None:
        self.posts = posts

    def resolve_category_id(self, slug: str) -> int:
        return 56

    def iter_posts(
        self, category_id: int, *, max_pages: int, per_page: int = 100
    ) -> Iterator[dict[str, Any]]:
        yield from self.posts


@pytest.fixture
def raw_posts() -> list[dict[str, Any]]:
    posts: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return posts


def test_output_path_flattens_nested_slugs(tmp_path: Path) -> None:
    assert output_path(tmp_path, "guardian/quick-cryptic/") == (
        tmp_path / "guardian_quick-cryptic.jsonl"
    )


def test_scrape_writes_posts_newest_first(tmp_path: Path, raw_posts: list[dict[str, Any]]) -> None:
    out = tmp_path / "qc.jsonl"

    summary = scrape_category(FakeWordPress(raw_posts[::-1]), "qc", max_pages=1, out_path=out)

    assert (summary.fetched, summary.new, summary.total) == (2, 2, 2)
    assert [p.id for p in read_jsonl(out, RawPost)] == [211874, 211645]


def test_rescrape_merges_instead_of_overwriting(
    tmp_path: Path, raw_posts: list[dict[str, Any]]
) -> None:
    out = tmp_path / "qc.jsonl"
    older, newer = raw_posts[1], raw_posts[0]
    scrape_category(FakeWordPress([older]), "qc", max_pages=1, out_path=out)

    summary = scrape_category(FakeWordPress([newer, older]), "qc", max_pages=1, out_path=out)

    assert (summary.fetched, summary.new, summary.total) == (2, 1, 2)
    assert len(read_jsonl(out, RawPost)) == 2  # no duplicate of `older`


def test_rescrape_keeps_posts_no_longer_fetched(
    tmp_path: Path, raw_posts: list[dict[str, Any]]
) -> None:
    out = tmp_path / "qc.jsonl"
    scrape_category(FakeWordPress(raw_posts), "qc", max_pages=1, out_path=out)

    summary = scrape_category(FakeWordPress(raw_posts[:1]), "qc", max_pages=1, out_path=out)

    assert summary.total == 2
