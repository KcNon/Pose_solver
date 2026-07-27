"""Small content-fingerprinted checkpoints for orchestration stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from common.io_utils import load_json, write_json


def stage_fingerprint(
    *,
    command: Iterable[str],
    content_files: Iterable[str | Path] = (),
    stat_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    digest = hashlib.sha256()
    normalized_command = [str(value) for value in command]
    digest.update(json.dumps(
        normalized_command,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8"))
    missing = []
    for value in sorted(Path(path) for path in content_files):
        name = str(value.resolve())
        digest.update(b"\0content\0" + name.encode("utf-8") + b"\0")
        if not value.exists():
            digest.update(b"missing")
            missing.append(name)
            continue
        with value.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    for value in sorted(Path(path) for path in stat_paths):
        name = str(value.resolve())
        digest.update(b"\0stat\0" + name.encode("utf-8") + b"\0")
        if not value.exists():
            digest.update(b"missing")
            missing.append(name)
            continue
        entries = (
            [value]
            if value.is_file()
            else sorted(path for path in value.rglob("*") if path.is_file())
        )
        for entry in entries:
            stat = entry.stat()
            relative = (
                entry.name
                if value.is_file()
                else str(entry.relative_to(value))
            )
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(
                f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii")
            )
    return {
        "algorithm": "sha256-command-content-stat-v1",
        "sha256": digest.hexdigest(),
        "command": normalized_command,
        "missing": missing,
    }


def checkpoint_path(expected: Path) -> Path:
    return expected.with_name(expected.name + ".checkpoint.json")


def checkpoint_matches(expected: Path, fingerprint: dict[str, Any]) -> bool:
    marker = checkpoint_path(expected)
    if not expected.exists() or not marker.exists():
        return False
    cached = load_json(marker)
    return cached.get("fingerprint", {}).get("sha256") == fingerprint["sha256"]


def write_checkpoint(
    expected: Path,
    fingerprint: dict[str, Any],
) -> None:
    write_json(checkpoint_path(expected), {
        "output": str(expected),
        "fingerprint": fingerprint,
    })
