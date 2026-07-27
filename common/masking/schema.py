"""Configuration schema for reusable multi-part mask extraction."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


LEGACY_IDS = {"lid": 1, "body": 2, "inner_pot": 3}
LEGACY_COLORS = {
    "lid": (255, 59, 48),
    "body": (52, 199, 89),
    "inner_pot": (0, 122, 255),
}
AUTO_COLORS = (
    (255, 59, 48),
    (52, 199, 89),
    (0, 122, 255),
    (255, 149, 0),
    (175, 82, 222),
    (90, 200, 250),
    (255, 204, 0),
    (0, 199, 190),
)


@dataclass(frozen=True)
class PartSpec:
    name: str
    id: int
    color: tuple[int, int, int]
    start_frame: int
    prompts: tuple[str, ...]
    start_frame_auto: bool = False


@dataclass(frozen=True)
class MaskPipelineConfig:
    source_path: Path
    raw: dict[str, Any]
    frames_dir: Path
    work_root: Path
    output_root: Path
    views: tuple[str, ...]
    parts: tuple[PartSpec, ...]
    occlusion_order: tuple[str, ...]

    @property
    def part_names(self) -> list[str]:
        return [part.name for part in self.parts]

    @property
    def part_map(self) -> dict[str, PartSpec]:
        return {part.name: part for part in self.parts}

    @property
    def tracks_root(self) -> Path:
        return self.work_root / "tracks"

    @property
    def bbox_path(self) -> Path:
        configured = self.raw.get("bbox_json")
        if configured:
            return Path(configured)
        if self.raw.get("masks_dir"):
            return Path(self.raw["masks_dir"]) / "bbox.json"
        return self.work_root / "bboxes" / "bbox.json"

    @property
    def masks_root(self) -> Path:
        return self.output_root / "masks"


def _color(value: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    selected = fallback if value is None else value
    if not isinstance(selected, (list, tuple)) or len(selected) != 3:
        raise ValueError(f"part color must be an RGB triplet, got {selected!r}")
    result = tuple(int(channel) for channel in selected)
    if any(channel < 0 or channel > 255 for channel in result):
        raise ValueError(f"part color channels must be in [0,255], got {result}")
    return result


def _part_entries(raw: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    configured = raw.get("parts")
    if isinstance(configured, dict):
        return [
            (str(name), dict(value or {}))
            for name, value in configured.items()
        ]
    if isinstance(configured, list) and configured:
        return [(str(name), {}) for name in configured]
    legacy_names = list(raw.get("part_ids", LEGACY_IDS))
    return [(str(name), {}) for name in legacy_names]


def _parts(raw: dict[str, Any]) -> tuple[PartSpec, ...]:
    entries = _part_entries(raw)
    ids = raw.get("part_ids", {})
    colors = raw.get("part_colors", {})
    starts = raw.get("part_start_frames", {})
    prompts = raw.get("prompts", {})
    reserved_legacy_ids = {
        LEGACY_IDS[name] for name, _values in entries if name in LEGACY_IDS
    }
    used_ids: set[int] = set()
    result: list[PartSpec] = []
    for index, (name, values) in enumerate(entries):
        configured_id = values.get("id", ids.get(name))
        if configured_id is not None:
            part_id = int(configured_id)
        elif name in LEGACY_IDS:
            part_id = LEGACY_IDS[name]
        else:
            default_id = next(
                candidate for candidate in range(1, 256)
                if candidate not in used_ids
                and candidate not in reserved_legacy_ids
            )
            part_id = default_id
        if part_id <= 0 or part_id >= 256:
            raise ValueError(f"part {name!r} id must be in [1,255], got {part_id}")
        if part_id in used_ids:
            raise ValueError(f"duplicate part id {part_id}")
        used_ids.add(part_id)
        fallback_color = LEGACY_COLORS.get(name, AUTO_COLORS[index % len(AUTO_COLORS)])
        part_color = _color(values.get("color", colors.get(name)), fallback_color)
        configured_start = values.get("start_frame", starts.get(name, 0))
        start_frame_auto = (
            isinstance(configured_start, str)
            and configured_start.strip().lower() == "auto"
        )
        if start_frame_auto:
            # An unresolved automatic start must stay active during the sparse
            # Qwen discovery pass.  The runner writes an integer start to its
            # resolved config before SAM or composition is allowed to run.
            start_frame = 0
        else:
            start_frame = int(configured_start)
        if start_frame < 0:
            raise ValueError(f"part {name!r} start_frame must be non-negative")
        configured_prompts = values.get("prompts", prompts.get(name))
        if configured_prompts is None:
            configured_prompts = [name.replace("_", " ")]
        if isinstance(configured_prompts, str):
            configured_prompts = [configured_prompts]
        prompt_tuple = tuple(str(prompt).strip() for prompt in configured_prompts if str(prompt).strip())
        if not prompt_tuple:
            raise ValueError(f"part {name!r} must have at least one prompt")
        result.append(PartSpec(
            name,
            part_id,
            part_color,
            start_frame,
            prompt_tuple,
            start_frame_auto,
        ))
    if not result:
        raise ValueError("at least one part is required")
    return tuple(result)


def _occlusion_order(raw: dict[str, Any], names: list[str]) -> tuple[str, ...]:
    configured = raw.get("occlusion_order") or raw.get("visibility_order_front_to_back")
    if configured is None:
        legacy = [name for name in ("lid", "inner_pot", "body") if name in names]
        configured = legacy + [name for name in names if name not in legacy]
    order = tuple(str(name) for name in configured)
    if len(order) != len(set(order)):
        raise ValueError(f"duplicate names in occlusion_order: {order}")
    unknown = set(order).difference(names)
    missing = set(names).difference(order)
    if unknown or missing:
        raise ValueError(
            f"occlusion_order must contain every part exactly once; "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return order


def load_mask_pipeline_config(path: str | Path) -> MaskPipelineConfig:
    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    frames_dir = Path(raw["frames_dir"])
    legacy_masks = Path(raw["masks_dir"]) if raw.get("masks_dir") else None
    work_default = (
        legacy_masks.parent / "mask_tracks"
        if legacy_masks is not None else source.parent / "mask_work"
    )
    work_root = Path(raw.get("work_root", work_default))
    output_default = legacy_masks.parent if legacy_masks is not None else work_root / "output"
    output_root = Path(raw.get("output_root", output_default))
    views = tuple(str(view) for view in raw.get(
        "views", ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6"]
    ))
    if len(views) != len(set(views)):
        raise ValueError(f"duplicate view names: {views}")
    parts = _parts(raw)
    order = _occlusion_order(raw, [part.name for part in parts])
    return MaskPipelineConfig(
        source_path=source,
        raw=raw,
        frames_dir=frames_dir,
        work_root=work_root,
        output_root=output_root,
        views=views,
        parts=parts,
        occlusion_order=order,
    )
