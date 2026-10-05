import hashlib
from pathlib import Path

import pytest
import responses

from cryptic_agent.lexicon.sources import ChecksumMismatchError, Source, fetch, sha256_of

CONTENT = b"word list contents"


def make_source(content: bytes = CONTENT) -> Source:
    return Source(
        name="test",
        url="https://example.com/words.zip",
        sha256=hashlib.sha256(content).hexdigest(),
        filename="words.zip",
        licence="test licence",
        homepage="https://example.com",
    )


@responses.activate
def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    responses.get("https://example.com/words.zip", body=CONTENT)

    path = fetch(make_source(), tmp_path / "sources")

    assert path.read_bytes() == CONTENT
    assert sha256_of(path) == make_source().sha256


@responses.activate
def test_fetch_skips_a_verified_existing_file(tmp_path: Path) -> None:
    (tmp_path / "words.zip").write_bytes(CONTENT)

    fetch(make_source(), tmp_path)

    assert len(responses.calls) == 0


@responses.activate
def test_fetch_replaces_a_corrupted_existing_file(tmp_path: Path) -> None:
    (tmp_path / "words.zip").write_bytes(b"truncated")
    responses.get("https://example.com/words.zip", body=CONTENT)

    assert fetch(make_source(), tmp_path).read_bytes() == CONTENT


@responses.activate
def test_checksum_mismatch_raises_and_leaves_nothing_behind(tmp_path: Path) -> None:
    responses.get("https://example.com/words.zip", body=b"changed upstream")

    with pytest.raises(ChecksumMismatchError, match="expected"):
        fetch(make_source(), tmp_path)

    assert list(tmp_path.iterdir()) == []  # neither the file nor a .part leftover


@responses.activate
def test_http_errors_propagate(tmp_path: Path) -> None:
    responses.get("https://example.com/words.zip", status=404)

    with pytest.raises(Exception, match="404"):
        fetch(make_source(), tmp_path)

    assert list(tmp_path.iterdir()) == []
