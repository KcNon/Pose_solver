#!/usr/bin/env python
"""Transfer an articulated two-link split to another hinge state by geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"expected a non-empty mesh: {path}")
    return mesh


def weighted_upper_quantile(
    values: np.ndarray, weights: np.ndarray, upper_fraction: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    target = (1.0 - upper_fraction) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target)), len(values) - 1)
    return float(sorted_values[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-mesh", type=Path, required=True)
    parser.add_argument("--reference-head", type=Path, required=True)
    parser.add_argument("--reference-body", type=Path, required=True)
    parser.add_argument("--hinge-prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state = load_mesh(args.state_mesh)
    reference_head = load_mesh(args.reference_head)
    reference_body = load_mesh(args.reference_body)
    prior = json.loads(args.hinge_prior.read_text(encoding="utf-8"))
    pivot = np.asarray(prior["point_close_raw"], dtype=np.float64)

    centered = np.asarray(state.vertices, dtype=np.float64) - state.centroid
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    long_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    if np.dot(pivot - state.centroid, long_axis) < 0:
        long_axis = -long_axis

    target_head_fraction = float(
        reference_head.area / (reference_head.area + reference_body.area)
    )
    scores = np.asarray(state.triangles_center) @ long_axis
    threshold = weighted_upper_quantile(
        scores, np.asarray(state.area_faces), target_head_fraction
    )
    head_mask = scores >= threshold
    body_mask = ~head_mask
    output_dir = args.output_dir.resolve()
    outputs = {}
    for name, selected in (("head", head_mask), ("body", body_mask)):
        part = state.submesh([selected], append=True, repair=False)
        part_dir = output_dir / name
        part_dir.mkdir(parents=True, exist_ok=True)
        path = part_dir / "model.glb"
        part.export(path)
        outputs[name] = {
            "mesh": str(path),
            "faces": int(selected.sum()),
            "face_fraction": float(selected.mean()),
            "surface_area_fraction": float(state.area_faces[selected].sum() / state.area),
            "bounds": np.asarray(part.bounds).tolist(),
        }

    labels = np.where(head_mask, 0, 1)
    review = state.copy()
    palette = np.asarray([[255, 59, 48, 255], [52, 199, 89, 255]], dtype=np.uint8)
    review.visual = trimesh.visual.ColorVisuals(
        mesh=review, face_colors=palette[labels]
    )
    review.export(output_dir / "parts_colored.glb")
    report = {
        "schema_version": 1,
        "method": "principal_long_axis_area_matched_hinge_state_transfer",
        "state_mesh": str(args.state_mesh.resolve()),
        "reference_parts": {
            "head": str(args.reference_head.resolve()),
            "body": str(args.reference_body.resolve()),
        },
        "target_head_surface_area_fraction": target_head_fraction,
        "long_axis_state_raw": long_axis.tolist(),
        "hinge_point_state_raw": pivot.tolist(),
        "hinge_projection": float(pivot @ long_axis),
        "split_projection": threshold,
        "split_to_hinge_distance": float(pivot @ long_axis - threshold),
        "parts": outputs,
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
