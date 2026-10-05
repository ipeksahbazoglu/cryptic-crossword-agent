"""Third-party data the solver relies on, downloaded reproducibly.

Each source is pinned to a URL and a SHA-256 checksum: if a download is
corrupted, truncated, or the publisher changes the file, the checksum won't
match and ingest stops instead of silently building from different data.

Downloads go to the gitignored data/ directory and are never committed: the
repo is public and some licences (e.g. ODbL) attach conditions to
redistributing derived databases.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import requests

from cryptic_agent.scraper.client import make_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    filename: str
    licence: str
    homepage: str


UKACD = Source(
    name="ukacd",
    # The UK Advanced Cryptics Dictionary's own site is gone; this is the copy
    # bundled in the ccxxv package on PyPI (wordlists/UKACD.txt inside the zip).
    url=(
        "https://files.pythonhosted.org/packages/c1/88/"
        "86fc6f1bf3cec188e05aa89f2886a0b4a5f13ab50fa29df0ce8b65a66561/ccxxv-0.1.4.zip"
    ),
    sha256="6665f7bb3dda939c210f4c4f85042baca698879cbf3c1308f6495ab6cf1570a1",
    filename="ccxxv-0.1.4.zip",
    licence="BSD-style, (c) 2009 J Ross Beresford; keep the copyright notice",
    homepage="https://pypi.org/project/ccxxv/",
)

MOBY = Source(
    name="moby",
    url="https://www.gutenberg.org/files/3202/files/mthesaur.txt",
    sha256="7c9742b1ed94435a893c0719b426725edb8a5242f8c526a75461bd6cee2dfd32",
    filename="mthesaur.txt",
    licence="Public domain (Grady Ward's Moby Thesaurus II, via Project Gutenberg)",
    homepage="https://www.gutenberg.org/ebooks/3202",
)

CRYPTICS = Source(
    name="cryptics",
    url="https://cryptics.georgeho.org/data.db",
    sha256="947f8992abb249533ce3ca0d73754f864a2ad8a08e31322777ec6e6aa483ede1",
    filename="cryptics.db",
    licence=(
        "ODbL v1.0 (George Ho, cryptics.georgeho.org). Derived databases we publish "
        "must also be ODbL, so the built lexicon is never committed"
    ),
    homepage="https://cryptics.georgeho.org/",
)

SOURCES: dict[str, Source] = {s.name: s for s in (UKACD, MOBY, CRYPTICS)}


class ChecksumMismatchError(RuntimeError):
    """A downloaded file is not the exact file we pinned."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: Source, dest_dir: Path, *, session: requests.Session | None = None) -> Path:
    """Download `source` into `dest_dir` (once) and return the verified file's path."""
    path = dest_dir / source.filename
    if path.exists() and sha256_of(path) == source.sha256:
        logger.info("%s: already downloaded", source.name)
        return path

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.part")
    digest = hashlib.sha256()
    logger.info("%s: downloading %s", source.name, source.url)
    try:
        with (session or make_session()).get(source.url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != source.sha256:
            raise ChecksumMismatchError(
                f"{source.name}: checksum mismatch for {source.url}\n"
                f"  expected {source.sha256}\n  got      {digest.hexdigest()}\n"
                "The file changed upstream or the download was corrupted."
            )
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path
