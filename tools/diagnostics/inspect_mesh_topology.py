#!/usr/bin/env python3
"""Write bounded connected-component and axis statistics for one mesh.

This diagnostic deliberately uses sparse face labels instead of
``Trimesh.split()`` so dense reconstruction meshes are never duplicated.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.io_utils import write_json
from common.resource_safety import require_memory_guard
from common.simulation_assets import load_flat_mesh


MAX_FACES = 1_000_000


def main() -> None:
    require_memory_guard("tools/diagnostics/inspect_mesh_topology.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-faces", type=int, default=MAX_FACES)
    args = parser.parse_args()

    mesh_path = args.mesh.expanduser().resolve()
    mesh = load_flat_mesh(mesh_path)
    if len(mesh.faces) > int(args.maximum_faces):
        raise RuntimeError(
            f"refusing {len(mesh.faces)} faces; limit={args.maximum_faces}"
        )
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    counts = np.bincount(labels)
    areas = np.bincount(
        labels,
        weights=np.asarray(mesh.area_faces, dtype=np.float64),
        minlength=len(counts),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rows = []
    for label in np.argsort(areas)[::-1]:
        face_ids = np.flatnonzero(labels == label)
        vertex_ids = np.unique(faces[face_ids].reshape(-1))
        points = vertices[vertex_ids]
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        extents = bounds[1] - bounds[0]
        rows.append({
            "label": int(label),
            "faces": int(len(face_ids)),
            "face_fraction": float(len(face_ids) / len(faces)),
            "area": float(areas[label]),
            "area_fraction": float(areas[label] / max(float(areas.sum()), 1e-12)),
            "vertices": int(len(vertex_ids)),
            "bounds": bounds.tolist(),
            "extents": extents.tolist(),
            "elongation": float(np.max(extents) / max(float(np.partition(extents, 1)[1]), 1e-12)),
        })
    quantiles = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
    report = {
        "schema_version": 1,
        "mesh": str(mesh_path),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "body_count": int(mesh.body_count),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
        "extents": np.asarray(mesh.extents, dtype=float).tolist(),
        "axis_quantiles": {
            axis: {
                str(value): float(np.quantile(vertices[:, index], value))
                for value in quantiles
            }
            for index, axis in enumerate("xyz")
        },
        "components": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"mesh topology -> {args.output}")


if __name__ == "__main__":
    main()
