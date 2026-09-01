#!/usr/bin/env python3
"""Inspect bounded reconstruction camera metadata across frame NPZ files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


MAX_FRAMES = 1_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    return parser


def _camera_key(files: list[str], singular: str, plural: str) -> str:
    if singular in files:
        return singular
    if plural in files:
        return plural
    raise KeyError(f"neither {singular!r} nor {plural!r} is present")


def _load_camera(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...] | None, str | None, list[str]]:
    with np.load(path, allow_pickle=False) as values:
        files = list(values.files)
        intrinsic_key = _camera_key(files, "intrinsic", "intrinsics")
        extrinsic_key = _camera_key(files, "extrinsic", "extrinsics")
        intrinsics = np.asarray(values[intrinsic_key], dtype=np.float64)
        extrinsics = np.asarray(values[extrinsic_key], dtype=np.float64)
        views = (
            tuple(str(value) for value in values["view_names"].tolist())
            if "view_names" in files
            else None
        )
        camera_mode = str(values["camera_mode"].item()) if "camera_mode" in files else None
    return intrinsics, extrinsics, views, camera_mode, files


def _camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return -np.einsum("vji,vj->vi", rotation, translation)


def _rotation_angles_degrees(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    relative = np.einsum(
        "vij,vkj->vik", current[:, :3, :3], reference[:, :3, :3]
    )
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def main() -> None:
    args = _parser().parse_args()
    if not 1 <= args.max_frames <= MAX_FRAMES:
        raise ValueError(f"max-frames must be in [1, {MAX_FRAMES}]")
    root = args.root.resolve()
    paths = sorted(root.glob("[0-9][0-9][0-9][0-9][0-9][0-9]/predictions.npz"))
    if not paths:
        raise FileNotFoundError(f"no frame prediction files under {root}")
    if len(paths) > args.max_frames:
        raise ValueError(f"found {len(paths)} frames, exceeding limit {args.max_frames}")

    reference_k, reference_e, reference_views, reference_mode, keys = _load_camera(paths[0])
    max_k = 0.0
    max_e = 0.0
    max_k_frame = paths[0].parent.name
    max_e_frame = paths[0].parent.name
    exact_k = True
    exact_e = True
    mismatched_views: list[str] = []
    camera_modes = {reference_mode}
    view_count = int(reference_e.shape[0])
    max_center_delta = np.zeros(view_count, dtype=np.float64)
    max_rotation_delta = np.zeros(view_count, dtype=np.float64)
    reference_centers = _camera_centers(reference_e)

    for path in paths[1:]:
        current_k, current_e, current_views, current_mode, _ = _load_camera(path)
        camera_modes.add(current_mode)
        if current_k.shape != reference_k.shape:
            raise ValueError(f"intrinsic shape mismatch at {path}: {current_k.shape} != {reference_k.shape}")
        if current_e.shape != reference_e.shape:
            raise ValueError(f"extrinsic shape mismatch at {path}: {current_e.shape} != {reference_e.shape}")
        if current_views != reference_views:
            mismatched_views.append(path.parent.name)
        exact_k = exact_k and np.array_equal(current_k, reference_k, equal_nan=True)
        exact_e = exact_e and np.array_equal(current_e, reference_e, equal_nan=True)
        delta_k = float(np.nanmax(np.abs(current_k - reference_k)))
        delta_e = float(np.nanmax(np.abs(current_e - reference_e)))
        if delta_k > max_k:
            max_k, max_k_frame = delta_k, path.parent.name
        if delta_e > max_e:
            max_e, max_e_frame = delta_e, path.parent.name
        max_center_delta = np.maximum(
            max_center_delta,
            np.linalg.norm(_camera_centers(current_e) - reference_centers, axis=1),
        )
        max_rotation_delta = np.maximum(
            max_rotation_delta,
            _rotation_angles_degrees(reference_e, current_e),
        )

    print(f"root={root}")
    print(f"frames={len(paths)} range={paths[0].parent.name}-{paths[-1].parent.name}")
    print(f"npz_keys={','.join(keys)}")
    print(f"intrinsics_shape={reference_k.shape} extrinsics_shape={reference_e.shape}")
    print(f"view_names={reference_views}")
    print(f"camera_modes={sorted(str(value) for value in camera_modes)}")
    print(f"view_names_fixed={not mismatched_views} mismatched_frames={mismatched_views}")
    print(f"intrinsics_exactly_fixed={exact_k} max_abs_delta={max_k:.17g} frame={max_k_frame}")
    print(f"extrinsics_exactly_fixed={exact_e} max_abs_delta={max_e:.17g} frame={max_e_frame}")
    names = reference_views or tuple(str(index) for index in range(view_count))
    for index, name in enumerate(names):
        print(
            f"view={name} max_camera_center_delta={max_center_delta[index]:.17g} "
            f"max_rotation_delta_deg={max_rotation_delta[index]:.17g}"
        )


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common.resource_safety import require_memory_guard

    require_memory_guard("tools/diagnostics/inspect_recon_cameras.py")
    main()
