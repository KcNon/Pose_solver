"""Small, dependency-free helpers for project JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from ``path``."""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    """Write readable UTF-8 JSON, creating parent directories as needed."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
