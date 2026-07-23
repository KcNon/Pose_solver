"""Trajectory serialization shared by solver and refinements."""
from __future__ import annotations

import csv
from pathlib import Path


def write_trajectory_csv(trajectory: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "frame", "part", "state", "source", "observing_views",
            "tx", "ty", "tz", "qx", "qy", "qz", "qw",
            "translation_step_m", "rotation_step_deg",
        ])
        for key, frame in trajectory["frames"].items():
            for part in trajectory["parts"]:
                record = frame["parts"][part]
                writer.writerow([
                    int(key), part, record["state"], record["source"],
                    record["observing_views"], *record["translation_body_m"],
                    *record["quaternion_body_xyzw"], record["translation_step_m"],
                    record["rotation_step_deg"],
                ])

