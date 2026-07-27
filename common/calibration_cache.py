from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from common.mask_io import frame_path
from common.normalized_recon import recon_npz_path

CALIBRATION_ALGORITHM_REVISION = "geometry-appearance-symmetry-fixed-scale-v5"


def _update_file_content(digest: Any, path: Path, logical_name: str) -> None:
    digest.update(b"content\0")
    digest.update(logical_name.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)


def _update_file_stat(digest: Any, path: Path, logical_name: str) -> None:
    stat = path.stat()
    digest.update(b"stat\0")
    digest.update(logical_name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(b"\0")


def fingerprint_files(
    *,
    config: dict[str, Any],
    content_files: Iterable[tuple[str, Path]],
    stat_files: Iterable[tuple[str, Path]],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    canonical_config = json.dumps(
        {
            "algorithm_revision": CALIBRATION_ALGORITHM_REVISION,
            "config": config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(b"config\0")
    digest.update(canonical_config)

    content_rows = sorted((name, Path(path)) for name, path in content_files)
    stat_rows = sorted((name, Path(path)) for name, path in stat_files)
    missing: list[str] = []
    for name, path in content_rows:
        if path.exists():
            _update_file_content(digest, path, name)
        else:
            digest.update(f"missing-content\0{name}\0".encode("utf-8"))
            missing.append(name)
    for name, path in stat_rows:
        if path.exists():
            _update_file_stat(digest, path, name)
        else:
            digest.update(f"missing-stat\0{name}\0".encode("utf-8"))
            missing.append(name)
    return {
        "algorithm": "sha256-config-content-and-stat-v1",
        "algorithm_revision": CALIBRATION_ALGORITHM_REVISION,
        "sha256": digest.hexdigest(),
        "content_file_count": len(content_rows),
        "stat_file_count": len(stat_rows),
        "missing": missing,
    }


def _geometry_frames(cfg: dict[str, Any], part: str) -> list[int]:
    state = cfg["states"][part]
    if part == cfg["reference_part"]:
        return sorted({int(value) for value in state["calibration_frames"]})
    windows = state.get("anchor_windows", {})
    frames: set[int] = set()
    for anchor in state["anchor_frames"]:
        frames.update(int(value) for value in windows.get(str(anchor), [anchor]))
    return sorted(frames)


def _appearance_frames(state: dict[str, Any], geometry_frames: list[int]) -> list[int]:
    appearance = state.get("appearance", {})
    if not appearance.get("enabled", False):
        return []
    evidence = appearance.get("anchor_evidence_frames", {})
    values: set[int] = set()
    for frame_list in evidence.values():
        if isinstance(frame_list, int):
            values.add(int(frame_list))
        else:
            values.update(int(value) for value in frame_list)
    if not values:
        values.update(geometry_frames)
    return sorted(values)


def build_calibration_fingerprint(
    cfg: dict[str, Any],
    *,
    cloud_root: Path,
    mesh_dir: Path,
) -> dict[str, Any]:
    """Fingerprint every direct calibration input without hashing large DA3 NPZs."""

    content_files: list[tuple[str, Path]] = []
    stat_files: list[tuple[str, Path]] = []
    for part in cfg["parts"]:
        content_files.append((f"mesh/{part}", mesh_dir / f"{part}.glb"))
        geometry_frames = _geometry_frames(cfg, part)
        for frame in geometry_frames:
            content_files.append(
                (
                    f"cloud/{part}/{frame:06d}",
                    cloud_root / f"{frame:06d}" / f"{part}.ply",
                )
            )
        state = cfg["states"][part]
        appearance = state.get("appearance", {})
        appearance_views = [
            str(view) for view in appearance.get("views", cfg["views"])
        ]
        for frame in _appearance_frames(state, geometry_frames):
            stat_files.append(
                (
                    f"recon/{frame:06d}",
                    Path(recon_npz_path(cfg, f"{frame:06d}", cfg["recon_backend"])),
                )
            )
            for view in appearance_views:
                stat_files.append(
                    (
                        f"image/{view}/{frame:06d}",
                        Path(
                            frame_path(
                                cfg["frames_dir"],
                                cfg.get("frames_layout", "normalized"),
                                f"{frame:06d}",
                                view,
                            )
                        ),
                    )
                )
                stat_files.append(
                    (
                        f"mask/{part}/{view}/{frame:06d}",
                        Path(cfg["masks_dir"]) / f"{frame:06d}" / f"{view}.png",
                    )
                )
    calibration_config = {
        "views": cfg["views"],
        "parts": cfg["parts"],
        "part_ids": cfg["part_ids"],
        "reference_part": cfg["reference_part"],
        "recon_backend": cfg["recon_backend"],
        "states": {
            part: {
                key: cfg["states"][part][key]
                for key in (
                    "scale_prior",
                    "calibration_frames",
                    "anchor_frames",
                    "anchor_windows",
                    "symmetry",
                    "appearance",
                )
                if key in cfg["states"][part]
            }
            for part in cfg["parts"]
        },
    }
    return fingerprint_files(
        config=calibration_config,
        content_files=content_files,
        stat_files=stat_files,
    )
