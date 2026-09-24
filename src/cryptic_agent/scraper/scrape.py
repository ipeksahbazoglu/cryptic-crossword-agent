"""Scrape one category into a JSONL file, merging with what is already there."""

import logging
from dataclasses import dataclass
from pathlib import Path

from cryptic_agent.jsonl import read_jsonl, write_jsonl
from cryptic_agent.models import RawPost
from cryptic_agent.scraper.clean import clean_post
from cryptic_agent.scraper.client import WordPressClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScrapeSummary:
    fetched: int
    new: int
    total: int
    path: Path


def output_path(raw_dir: Path, category: str) -> Path:
    """'guardian/quick-cryptic' -> <raw_dir>/guardian_quick-cryptic.jsonl"""
    return raw_dir / f"{category.strip('/').replace('/', '_')}.jsonl"


def scrape_category(
    client: WordPressClient, category: str, *, max_pages: int, out_path: Path
) -> ScrapeSummary:
    """Fetch up to `max_pages` pages of `category` and merge them into `out_path`.

    Re-running is safe: posts are keyed by ID, so a later run adds new posts
    (and refreshes edited ones) instead of duplicating or discarding old ones.
    """
    existing = {p.id: p for p in read_jsonl(out_path, RawPost)} if out_path.exists() else {}

    category_id = client.resolve_category_id(category)
    logger.info("category %r -> id %d", category, category_id)

    fetched: dict[int, RawPost] = {}
    for raw in client.iter_posts(category_id, max_pages=max_pages):
        post = clean_post(raw)
        fetched[post.id] = post
    logger.info("fetched %d posts", len(fetched))

    merged = existing | fetched
    posts = sorted(merged.values(), key=lambda p: p.date, reverse=True)
    write_jsonl(out_path, posts)

    return ScrapeSummary(
        fetched=len(fetched),
        new=len(fetched.keys() - existing.keys()),
        total=len(posts),
        path=out_path,
    )
