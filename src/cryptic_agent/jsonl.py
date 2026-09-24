"""Read and write JSON Lines files of pydantic models (one JSON object per line)."""

import os
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, ValidationError


class JsonlError(ValueError):
    """A line in a JSONL file could not be parsed into the expected model."""


def iter_jsonl[M: BaseModel](path: Path, model: type[M]) -> Iterator[M]:
    """Yield one validated `model` per non-blank line of `path`.

    Errors name the file and line number, so a bad record in a large dataset
    is easy to find.
    """
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line)
            except ValidationError as exc:
                raise JsonlError(f"{path}:{line_number}: invalid {model.__name__}\n{exc}") from exc


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    """Load every record in `path` into memory."""
    return list(iter_jsonl(path, model))


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write `records` to `path`, replacing it, and return how many were written.

    Writes to a temporary file first and renames it into place, so a crash
    midway never leaves a half-written file where a complete one used to be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    count = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
                count += 1
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return count


def append_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Append `records` to `path` (creating it if needed) and return how many were written.

    For long-running jobs that save progress as they go, so a restart can pick
    up where it stopped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
            f.flush()
            count += 1
    return count
