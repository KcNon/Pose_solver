"""Frame, binary-track, bbox, and dynamic-palette mask I/O."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import PartSpec


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def frame_path(frames_dir: Path, view: str, timestamp: str) -> Path:
    for suffix in IMAGE_SUFFIXES:
        candidate = frames_dir / view / f"{timestamp}{suffix}"
        if candidate.exists():
            return candidate
    return frames_dir / view / f"{timestamp}.jpg"


def list_frame_ids(frames_dir: Path, view: str) -> list[str]:
    directory = frames_dir / view
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        (
            path.stem
            for path in directory.iterdir()
            if path.stem.isdigit() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=int,
    )


def validate_synchronized_frames(frames_dir: Path, views: Iterable[str]) -> list[str]:
    ordered = list(views)
    if not ordered:
        raise ValueError("at least one view is required")
    reference = list_frame_ids(frames_dir, ordered[0])
    for view in ordered[1:]:
        current = list_frame_ids(frames_dir, view)
        if current != reference:
            raise RuntimeError(f"timestamp sequence differs for {view}")
    return reference


def track_path(root: Path, part: str, timestamp: str, view: str) -> Path:
    return root / part / timestamp / f"{view}.png"


def validated_seed_path(
    work_root: Path,
    part: str,
    timestamp: str,
    view: str,
) -> Path:
    return work_root / "validated_seeds" / part / timestamp / f"{view}.png"


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=bool).astype(np.uint8) * 255, mode="L").save(path)


def load_binary_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    from PIL import Image

    if not path.exists():
        if shape is None:
            raise FileNotFoundError(path)
        return np.zeros(shape, dtype=bool)
    data = np.asarray(Image.open(path))
    if data.ndim != 2:
        raise ValueError(f"expected a single-channel mask: {path}")
    result = data > 0
    if shape is not None and result.shape != shape:
        raise ValueError(f"mask shape mismatch for {path}: {result.shape} != {shape}")
    return result


def load_label_mask(path: Path) -> np.ndarray:
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(path)
    label = np.asarray(Image.open(path))
    if label.ndim != 2:
        raise ValueError(f"expected indexed label mask: {path}")
    return label.astype(np.uint8, copy=False)


def build_palette(parts: Iterable[PartSpec]) -> list[int]:
    palette = [0] * (256 * 3)
    for part in parts:
        offset = part.id * 3
        palette[offset:offset + 3] = list(part.color)
    return palette


def save_label_mask(path: Path, label: np.ndarray, parts: Iterable[PartSpec]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.asarray(label, dtype=np.uint8), mode="P")
    image.putpalette(build_palette(parts))
    image.save(path, format="PNG", optimize=True)


def load_bbox_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"parts": {}, "frames": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_bbox_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
