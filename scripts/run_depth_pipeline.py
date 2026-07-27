#!/usr/bin/env python
"""Run fixed-rig DA3, depth-gauge calibration, and point-cloud extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.stage_cache import (
    checkpoint_matches,
    stage_fingerprint,
    write_checkpoint,
)

DEFAULT_CONFIG = ROOT / "configs" / "pipeline_data_1_8view.json"
STAGES = ROOT / "tools" / "stages" / "depth"


def _run(
    command: list[str],
    expected: Path,
    force: bool,
    *,
    content_files: tuple[Path, ...] = (),
    stat_paths: tuple[Path, ...] = (),
) -> None:
    fingerprint = stage_fingerprint(
        command=command,
        content_files=content_files,
        stat_paths=stat_paths,
    )
    if not force and checkpoint_matches(expected, fingerprint):
        print(f"[resume] {expected}", flush=True)
        return
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if not expected.exists():
        raise RuntimeError(
            f"stage completed without expected output: {expected}"
        )
    write_checkpoint(expected, fingerprint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--stage",
        choices=("all", "da3", "gauge", "cloud", "postprocess"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config.get("depth_pipeline", {})
    frames = config.get("frames", {})
    start = int(frames.get("start", 0))
    end = int(frames.get("end", 10**9))
    timestamps = [f"{frame:06d}" for frame in range(start, end + 1)]

    if args.stage in {"all", "da3"}:
        da3_output = Path(config["da3_self_cond_dir"]).resolve()
        expected = da3_output / timestamps[-1] / "predictions.npz"
        command = [
            str(runtime["da3_python"]),
            "-u",
            str(STAGES / "run_da3_fixed_rig.py"),
            "--frames-dir",
            str(Path(config["frames_dir"]).resolve()),
            "--output-dir",
            str(da3_output),
            "--views",
            *config["views"],
            "--timestamps",
            *timestamps,
            "--camera-frame",
            f"{int(runtime['camera_frame']):06d}",
            "--batch-size",
            str(int(runtime.get("batch_size", 1))),
            "--device",
            str(runtime.get("device", "cuda")),
            "--overwrite",
        ]
        _run(
            command,
            expected,
            args.force,
            stat_paths=(Path(config["frames_dir"]).resolve(),),
        )

    if args.stage in {"all", "gauge", "postprocess"}:
        gauge_path = Path(config["depth_gauge_path"]).resolve()
        command = [
            sys.executable,
            str(STAGES / "calibrate_depth_gauge.py"),
            "--pipeline",
            str(config_path),
            "--measure-start",
            str(int(runtime.get("gauge_measure_start", start))),
            "--measure-end",
            str(int(runtime.get("gauge_measure_end", end))),
            "--cross-view-stride",
            str(int(runtime.get("cross_view_stride", 5))),
            "--out",
            str(gauge_path),
        ]
        if runtime.get("cross_view", False):
            command.append("--cross-view")
        _run(
            command,
            gauge_path,
            args.force,
            content_files=(config_path,),
            stat_paths=(
                Path(config["masks_dir"]).resolve(),
                Path(config["da3_self_cond_dir"]).resolve(),
            ),
        )

    if args.stage in {"all", "cloud", "postprocess"}:
        backend = str(config["recon_backend"])
        tag = config.get("point_cloud_tag")
        name = backend if not tag else f"{backend}_{tag}"
        artifact_root = Path(
            config.get("point_cloud_output_root", config["output_root"])
        )
        expected = artifact_root / "parts_ply" / name / "backproject_summary.json"
        _run(
            [
                sys.executable,
                str(STAGES / "backproject_normalized.py"),
                "--pipeline",
                str(config_path),
                "--all",
                "--conf-mode",
                str(runtime.get("conf_mode", "adaptive")),
                "--conf-quantile",
                str(float(runtime.get("conf_quantile", 0.25))),
                "--stride",
                str(int(runtime.get("point_stride", 2))),
                "--max-pts",
                str(int(runtime.get("max_points", 80000))),
            ],
            expected,
            args.force,
            content_files=(
                config_path,
                Path(config["depth_gauge_path"]).resolve(),
            ),
            stat_paths=(
                Path(config["masks_dir"]).resolve(),
                Path(config["da3_self_cond_dir"]).resolve(),
            ),
        )


if __name__ == "__main__":
    main()
