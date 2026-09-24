from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from cryptic_agent.jsonl import JsonlError, append_jsonl, iter_jsonl, read_jsonl, write_jsonl
from cryptic_agent.models import RawPost


def make_post(post_id: int) -> RawPost:
    return RawPost(
        id=post_id,
        date=datetime(2026, 1, post_id),
        url=f"https://fifteensquared.net/{post_id}",
        title=f"Guardian {post_id}",
        content_markdown="**Senator** arranged crime",
    )


def test_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "posts.jsonl"
    posts = [make_post(1), make_post(2)]

    assert write_jsonl(path, posts) == 2
    assert read_jsonl(path, RawPost) == posts


def test_write_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "posts.jsonl"
    write_jsonl(path, [make_post(1), make_post(2)])

    write_jsonl(path, [make_post(3)])

    assert [p.id for p in read_jsonl(path, RawPost)] == [3]


def test_failed_write_keeps_previous_file(tmp_path: Path) -> None:
    path = tmp_path / "posts.jsonl"
    write_jsonl(path, [make_post(1)])

    def records_then_crash() -> Iterator[RawPost]:
        yield make_post(2)  # written to the temp file...
        raise RuntimeError("crashed mid-run")  # ...then the job dies

    with pytest.raises(RuntimeError, match="crashed mid-run"):
        write_jsonl(path, records_then_crash())

    assert [p.id for p in read_jsonl(path, RawPost)] == [1]
    assert list(tmp_path.iterdir()) == [path]  # temp file cleaned up


def test_append_adds_to_existing_records(tmp_path: Path) -> None:
    path = tmp_path / "posts.jsonl"
    append_jsonl(path, [make_post(1)])

    assert append_jsonl(path, [make_post(2)]) == 1
    assert [p.id for p in read_jsonl(path, RawPost)] == [1, 2]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "posts.jsonl"
    write_jsonl(path, [make_post(1)])
    path.write_text(path.read_text() + "\n  \n")

    assert len(read_jsonl(path, RawPost)) == 1


def test_invalid_line_reports_file_and_line_number(tmp_path: Path) -> None:
    path = tmp_path / "posts.jsonl"
    write_jsonl(path, [make_post(1)])
    path.write_text(path.read_text() + '{"id": "not-a-number"}\n')

    records = iter_jsonl(path, RawPost)
    assert next(records).id == 1
    with pytest.raises(JsonlError, match=r"posts\.jsonl:2: invalid RawPost"):
        next(records)
