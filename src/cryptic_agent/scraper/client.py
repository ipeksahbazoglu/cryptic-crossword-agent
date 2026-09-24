"""A polite client for the WordPress REST API that Fifteensquared exposes.

We use /wp-json/wp/v2 rather than scraping rendered pages: it is the site's own
public JSON API, so it doesn't break when the theme changes.

Politeness: at most one request per `min_interval` seconds, an identifying
User-Agent, and automatic retries with exponential backoff on 429/5xx
(honouring Retry-After).
"""

import time
from collections.abc import Callable, Iterator
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fifteensquared.net/wp-json/wp/v2"
USER_AGENT = "cryptic-agent/0.1 (+https://github.com/ipeksahbazoglu/cryptic-crossword-agent)"
MAX_PER_PAGE = 100  # WordPress's hard limit
POST_FIELDS = "id,date,link,title,content"


class CategoryNotFoundError(LookupError):
    """No WordPress category matches the given slug."""


def make_session() -> requests.Session:
    """A session with our User-Agent and retry policy."""
    retry = Retry(
        total=5,
        backoff_factor=2,  # waits 0s, 4s, 8s, 16s, 32s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class WordPressClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        session: requests.Session | None = None,
        min_interval: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or make_session()
        self._min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None

    def _get(self, path: str, params: dict[str, str | int]) -> requests.Response:
        if self._last_request is not None:
            wait = self._min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        try:
            return self._session.get(f"{self._base_url}{path}", params=params, timeout=30)
        finally:
            self._last_request = self._clock()

    def resolve_category_id(self, slug: str) -> int:
        """Numeric ID for a category slug.

        Fifteensquared's URLs nest categories ('guardian/quick-cryptic') but the
        API's slugs are flat, so only the last path segment is looked up.
        """
        short_slug = slug.strip("/").split("/")[-1]
        response = self._get("/categories", {"slug": short_slug})
        response.raise_for_status()
        matches: list[dict[str, Any]] = response.json()
        if not matches:
            raise CategoryNotFoundError(f"no Fifteensquared category with slug {short_slug!r}")
        return int(matches[0]["id"])

    def iter_posts(
        self, category_id: int, *, max_pages: int, per_page: int = MAX_PER_PAGE
    ) -> Iterator[dict[str, Any]]:
        """Yield raw post JSON, newest first, for up to `max_pages` pages."""
        for page in range(1, max_pages + 1):
            response = self._get(
                "/posts",
                {
                    "categories": category_id,
                    "page": page,
                    "per_page": per_page,
                    "_fields": POST_FIELDS,
                },
            )
            if response.status_code == 400:  # WordPress's answer to "past the last page"
                return
            response.raise_for_status()
            posts: list[dict[str, Any]] = response.json()
            yield from posts
            if page >= int(response.headers.get("X-WP-TotalPages", page)):
                return
