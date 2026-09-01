#!/usr/bin/env python
"""Fit a cylindrical connector axis and origin directly from a mesh slab."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.connector_geometry import fit_cylindrical_axis_from_slab
from common.io_utils import write_json
from common.resource_safety import require_memory_guard
from common.simulation_assets import load_flat_mesh


MAX_PREVIEW_POINTS = 50_000
MAX_FACES = 1_000_000


def _write_preview(
    path: Path,
    vertices: np.ndarray,
    roi: np.ndarray,
    report: dict,
) -> None:
    """Write bounded orthographic evidence for reviewing the selected slab."""

    stride = max(1, int(np.ceil(len(vertices) / MAX_PREVIEW_POINTS)))
    points = vertices[::stride]
    selected = roi[::stride]
    panels = []
    for horizontal, vertical, label in ((0, 2, "XZ"), (0, 1, "XY"), (2, 1, "ZY")):
        canvas = np.full((520, 520, 3), 248, dtype=np.uint8)
        values = points[:, [horizontal, vertical]]
        lower = np.quantile(values, 0.005, axis=0)
        upper = np.quantile(values, 0.995, axis=0)
        span = np.maximum(upper - lower, 1e-9)
        pixel = np.rint(24.0 + 472.0 * (values - lower) / span).astype(int)
        pixel = np.clip(pixel, 0, 519)
        canvas[519 - pixel[:, 1], pixel[:, 0]] = (190, 190, 190)
        chosen = pixel[selected]
        canvas[519 - chosen[:, 1], chosen[:, 0]] = (30, 120, 240)

        origin = np.asarray(report["origin_raw"], dtype=np.float64)
        axis = np.asarray(report["axis_part"], dtype=np.float64)
        line_scale = float(np.linalg.norm(upper - lower))
        endpoints = np.stack((origin - line_scale * axis, origin + line_scale * axis))
        projected = np.rint(
            24.0 + 472.0 * (endpoints[:, [horizontal, vertical]] - lower) / span
        ).astype(int)
        cv2.line(
            canvas,
            tuple(projected[0] * np.array([1, -1]) + np.array([0, 519])),
            tuple(projected[1] * np.array([1, -1]) + np.array([0, 519])),
            (220, 40, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{label}: gray=mesh orange=ROI magenta=fit",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        panels.append(canvas)
    preview = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(f"failed to write connector preview: {path}")


def main() -> None:
    require_memory_guard("tools/diagnostics/fit_mesh_connector_frame.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument(
        "--selector-axis", required=True, choices=("x", "y", "z")
    )
    parser.add_argument("--minimum", required=True, type=float)
    parser.add_argument("--maximum", required=True, type=float)
    parser.add_argument("--direction-sign", required=True, type=float)
    parser.add_argument("--origin-coordinate", required=True, type=float)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--quantile", type=float, default=0.02)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--component-labels", nargs="+", type=int)
    for axis in "xyz":
        parser.add_argument(f"--{axis}-minimum", type=float)
        parser.add_argument(f"--{axis}-maximum", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    loaded = load_flat_mesh(mesh_path)
    if len(loaded.faces) > MAX_FACES:
        raise RuntimeError(
            f"refusing {len(loaded.faces)} faces; limit={MAX_FACES}"
        )
    if not len(loaded.vertices):
        raise ValueError(f"expected a non-empty triangle mesh: {mesh_path}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    roi = np.ones(len(vertices), dtype=bool)
    roi_bounds = {}
    if args.component_labels:
        face_labels = trimesh.graph.connected_component_labels(
            loaded.face_adjacency, node_count=len(loaded.faces)
        )
        requested = np.asarray(sorted(set(args.component_labels)), dtype=int)
        available = np.unique(face_labels)
        missing = requested[~np.isin(requested, available)]
        if len(missing):
            raise ValueError(f"unknown component labels: {missing.tolist()}")
        selected_faces = np.flatnonzero(np.isin(face_labels, requested))
        selected_vertices = np.unique(
            np.asarray(loaded.faces, dtype=np.int64)[selected_faces].reshape(-1)
        )
        roi[:] = False
        roi[selected_vertices] = True
    for index, axis in enumerate("xyz"):
        lower = getattr(args, f"{axis}_minimum")
        upper = getattr(args, f"{axis}_maximum")
        if lower is not None:
            roi &= vertices[:, index] >= float(lower)
        if upper is not None:
            roi &= vertices[:, index] <= float(upper)
        if lower is not None or upper is not None:
            roi_bounds[axis] = [lower, upper]
    report = fit_cylindrical_axis_from_slab(
        vertices[roi],
        selector_axis=args.selector_axis,
        minimum=args.minimum,
        maximum=args.maximum,
        direction_sign=args.direction_sign,
        origin_coordinate=args.origin_coordinate,
        bins=args.bins,
        quantile=args.quantile,
    )
    report["mesh"] = str(mesh_path)
    report["roi_bounds"] = roi_bounds
    report["component_labels"] = (
        sorted(set(args.component_labels)) if args.component_labels else None
    )
    report["roi_vertices"] = int(np.count_nonzero(roi))
    write_json(args.output, report)
    if args.preview is not None:
        _write_preview(args.preview.expanduser().resolve(), vertices, roi, report)
    print(f"connector frame -> {args.output.expanduser().resolve()}")
    print(f"axis_part={report['axis_part']}")
    print(f"origin_raw={report['origin_raw']}")


if __name__ == "__main__":
    main()
