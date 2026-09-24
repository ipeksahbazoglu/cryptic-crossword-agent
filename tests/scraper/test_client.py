from urllib.parse import parse_qs, urlparse

import pytest
import responses
from requests.adapters import HTTPAdapter

from cryptic_agent.scraper.client import (
    BASE_URL,
    USER_AGENT,
    CategoryNotFoundError,
    WordPressClient,
    make_session,
)


class FakeClock:
    """Time that only moves when something sleeps, so rate limiting is testable instantly."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(clock: FakeClock | None = None) -> WordPressClient:
    clock = clock or FakeClock()
    return WordPressClient(clock=clock, sleep=clock.sleep)


def query(call_index: int) -> dict[str, list[str]]:
    url = responses.calls[call_index].request.url
    assert url is not None
    return parse_qs(urlparse(url).query)


@responses.activate
def test_resolve_category_uses_last_slug_segment() -> None:
    responses.get(f"{BASE_URL}/categories", json=[{"id": 56, "slug": "quick-cryptic"}])

    assert make_client().resolve_category_id("guardian/quick-cryptic/") == 56
    assert query(0)["slug"] == ["quick-cryptic"]


@responses.activate
def test_unknown_category_raises() -> None:
    responses.get(f"{BASE_URL}/categories", json=[])

    with pytest.raises(CategoryNotFoundError, match="nope"):
        make_client().resolve_category_id("nope")


@responses.activate
def test_iter_posts_pages_until_total_pages() -> None:
    for page in (1, 2):
        responses.get(
            f"{BASE_URL}/posts",
            json=[{"id": page * 10}, {"id": page * 10 + 1}],
            headers={"X-WP-TotalPages": "2"},
        )

    posts = list(make_client().iter_posts(56, max_pages=5))

    assert [p["id"] for p in posts] == [10, 11, 20, 21]
    assert len(responses.calls) == 2  # stopped at the last page, not max_pages
    assert query(1)["page"] == ["2"]
    assert query(1)["categories"] == ["56"]


@responses.activate
def test_iter_posts_respects_max_pages() -> None:
    responses.get(f"{BASE_URL}/posts", json=[{"id": 1}], headers={"X-WP-TotalPages": "69"})

    assert len(list(make_client().iter_posts(56, max_pages=1))) == 1
    assert len(responses.calls) == 1


@responses.activate
def test_iter_posts_stops_on_400_past_last_page() -> None:
    responses.get(f"{BASE_URL}/posts", status=400, json={"code": "rest_post_invalid_page_number"})

    assert list(make_client().iter_posts(56, max_pages=3)) == []


@responses.activate
def test_requests_are_rate_limited() -> None:
    responses.get(f"{BASE_URL}/posts", json=[{"id": 1}], headers={"X-WP-TotalPages": "3"})
    clock = FakeClock()

    list(make_client(clock).iter_posts(56, max_pages=3))

    assert clock.sleeps == [1.0, 1.0]  # none before the first request, then one per gap


@responses.activate
def test_requests_identify_themselves() -> None:
    responses.get(f"{BASE_URL}/categories", json=[{"id": 7}])

    make_client().resolve_category_id("guardian")

    assert responses.calls[0].request.headers["User-Agent"] == USER_AGENT


def test_session_retries_rate_limits_and_server_errors() -> None:
    # `responses` replaces the transport, so it can't exercise urllib3's retries;
    # check the policy is configured instead.
    adapter = make_session().get_adapter("https://fifteensquared.net")
    assert isinstance(adapter, HTTPAdapter)
    retry = adapter.max_retries

    assert retry.total == 5
    assert {429, 500, 502, 503, 504} <= set(retry.status_forcelist or [])
    assert retry.respect_retry_after_header
