"""Single-source configuration contract for mask, depth, and pose.

The stage implementations still consume their focused resolved JSON files.
Those files are generated artifacts, never additional user-maintained input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


VALID_MODES = {"run", "reuse"}


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class PartConfig:
    name: str
    id: int
    mesh: Path
    prompts: tuple[str, ...]
    appearance_hint: int | str
    reference: bool = False

    def mask_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_frame": self.appearance_hint,
            "prompts": list(self.prompts),
        }


@dataclass(frozen=True)
class StageConfig:
    mode: str
    artifact: Path | None
    overrides: dict[str, Any]
    compatibility_config: Path | None = None


@dataclass(frozen=True)
class PipelineConfig:
    source_path: Path
    raw: dict[str, Any]
    dataset: str
    frames_dir: Path
    views: tuple[str, ...]
    frame_start: int
    frame_end: int
    parts: tuple[PartConfig, ...]
    output_root: Path
    devices: tuple[int, ...]
    models: dict[str, Any]
    input_options: dict[str, Any]
    mask: StageConfig
    depth: StageConfig
    pose: StageConfig

    @property
    def part_names(self) -> list[str]:
        return [part.name for part in self.parts]

    @property
    def part_ids(self) -> dict[str, int]:
        return {part.name: part.id for part in self.parts}

    @property
    def reference_part(self) -> str:
        selected = [part.name for part in self.parts if part.reference]
        return selected[0] if selected else self.parts[0].name

    @property
    def mesh_dir(self) -> Path:
        return self.parts[0].mesh.parent

    @property
    def videos(self) -> dict[str, Path]:
        return {
            str(view): Path(path)
            for view, path in self.input_options.get("videos", {}).items()
        }


def _stage(raw: Mapping[str, Any], name: str, base: Path) -> StageConfig:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    mode = str(value.get("mode", "run"))
    if mode not in VALID_MODES:
        raise ValueError(f"{name}.mode must be one of {sorted(VALID_MODES)}")
    artifact_value = value.get("artifact")
    artifact = _resolve_path(artifact_value, base) if artifact_value else None
    compatibility_value = value.get("compatibility_config")
    compatibility = (
        _resolve_path(compatibility_value, base)
        if compatibility_value
        else None
    )
    overrides = value.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{name}.overrides must be an object")
    if mode == "reuse" and artifact is None:
        raise ValueError(f"{name}.artifact is required when mode is 'reuse'")
    return StageConfig(mode, artifact, deepcopy(dict(overrides)), compatibility)


def _parts(raw: Mapping[str, Any], base: Path) -> tuple[PartConfig, ...]:
    configured = raw.get("parts")
    if not isinstance(configured, Mapping) or not configured:
        raise ValueError("parts must be a non-empty object")
    result: list[PartConfig] = []
    ids: set[int] = set()
    references = 0
    for name, value in configured.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"parts.{name} must be an object")
        part_id = int(value["id"])
        if not 1 <= part_id <= 255:
            raise ValueError(f"parts.{name}.id must be in [1, 255]")
        if part_id in ids:
            raise ValueError(f"duplicate part id {part_id}")
        ids.add(part_id)
        mesh = _resolve_path(value["mesh"], base)
        prompts_value = value.get("prompts", value.get("prompt", name))
        if isinstance(prompts_value, str):
            prompts_value = [prompts_value]
        prompts = tuple(
            str(item).strip()
            for item in prompts_value
            if str(item).strip()
        )
        if not prompts:
            raise ValueError(f"parts.{name} needs at least one prompt")
        appearance_hint = value.get("appearance_hint", "auto")
        if not (
            appearance_hint == "auto"
            or isinstance(appearance_hint, int)
            or (isinstance(appearance_hint, str) and appearance_hint.isdigit())
        ):
            raise ValueError(
                f"parts.{name}.appearance_hint must be an integer or 'auto'"
            )
        appearance_hint = (
            "auto" if appearance_hint == "auto" else int(appearance_hint)
        )
        reference = bool(value.get("reference", False))
        references += int(reference)
        result.append(PartConfig(
            str(name), part_id, mesh, prompts, appearance_hint, reference
        ))
    if references > 1:
        raise ValueError("at most one part may set reference=true")
    mesh_dirs = {part.mesh.parent for part in result}
    if len(mesh_dirs) != 1:
        raise ValueError(
            "all part meshes must share one directory; the pose solver uses "
            "the <mesh_dir>/<part>.glb contract"
        )
    for part in result:
        if part.mesh.name != f"{part.name}.glb":
            raise ValueError(
                f"parts.{part.name}.mesh must be named {part.name}.glb"
            )
    return tuple(result)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate the one user-maintained source config."""

    source = Path(path).resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", 1)) != 1:
        raise ValueError("only pipeline schema_version=1 is supported")
    base = source.parent
    dataset = str(raw.get("dataset", "")).strip()
    if not dataset:
        raise ValueError("dataset must be a non-empty string")
    input_raw = raw.get("input")
    if not isinstance(input_raw, Mapping):
        raise ValueError("input must be an object")
    frames_dir = _resolve_path(input_raw["frames_dir"], base)
    views = tuple(str(view) for view in input_raw.get("views", ()))
    if not views or len(views) != len(set(views)):
        raise ValueError("input.views must be non-empty and unique")
    frame_range = input_raw.get("frame_range")
    if not isinstance(frame_range, list) or len(frame_range) != 2:
        raise ValueError("input.frame_range must be [start, end]")
    frame_start, frame_end = map(int, frame_range)
    if frame_start < 0 or frame_end < frame_start:
        raise ValueError(f"invalid input.frame_range {frame_start}..{frame_end}")
    parts = _parts(raw, base)
    output_raw = raw.get("output", {})
    if not isinstance(output_raw, Mapping) or not output_raw.get("root"):
        raise ValueError("output.root is required")
    output_root = _resolve_path(output_raw["root"], base)
    runtime = raw.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be an object")
    devices = tuple(int(value) for value in runtime.get("devices", ()))
    if not devices:
        raise ValueError("runtime.devices must contain one or two GPU indices")
    if len(devices) > 2 or len(devices) != len(set(devices)):
        raise ValueError("runtime.devices must contain one or two unique GPUs")
    if any(device < 0 for device in devices):
        raise ValueError("runtime.devices cannot contain negative indices")
    models = raw.get("models", {})
    if not isinstance(models, Mapping):
        raise ValueError("models must be an object")
    resolved_models = deepcopy(dict(models))
    for key in (
        "qwen_python",
        "sam_python",
        "qwen_model",
        "sam_checkpoint",
        "da3_python",
    ):
        if resolved_models.get(key):
            resolved_models[key] = str(_resolve_path(resolved_models[key], base))
    input_options = deepcopy(dict(input_raw))
    for key in ("depth_dir", "videos_dir"):
        if input_options.get(key):
            input_options[key] = str(_resolve_path(input_options[key], base))
    videos = input_options.get("videos", {})
    if videos:
        if not isinstance(videos, Mapping):
            raise ValueError("input.videos must map every view to a video")
        if set(map(str, videos)) != set(views):
            raise ValueError("input.videos keys must exactly match input.views")
        input_options["videos"] = {
            str(view): str(_resolve_path(value, base))
            for view, value in videos.items()
        }
    return PipelineConfig(
        source_path=source,
        raw=raw,
        dataset=dataset,
        frames_dir=frames_dir,
        views=views,
        frame_start=frame_start,
        frame_end=frame_end,
        parts=parts,
        output_root=output_root,
        devices=devices,
        models=resolved_models,
        input_options=input_options,
        mask=_stage(raw, "mask", base),
        depth=_stage(raw, "depth", base),
        pose=_stage(raw, "pose", base),
    )
