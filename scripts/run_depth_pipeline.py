#!/usr/bin/env python
"""Run fixed-rig DA3, depth-gauge calibration, and point-cloud extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.stage_cache import (
    checkpoint_matches,
    stage_fingerprint,
    write_checkpoint,
)

STAGES = ROOT / "tools" / "stages" / "depth"


def _optional_path(value: str | None) -> tuple[Path, ...]:
    return (Path(value).resolve(),) if value else ()


def build_da3_command(
    config: dict,
    runtime: dict,
    timestamps: list[str],
    *,
    force: bool = False,
) -> list[str]:
    """Build the fixed-rig DA3 command from explicit, fingerprinted settings."""

    process_res = int(runtime.get("process_res", 504))
    full_w = int(runtime.get("full_w", 1920))
    full_h = int(runtime.get("full_h", 1080))
    if process_res <= 0:
        raise ValueError("depth_pipeline.process_res must be positive")
    if full_w <= 0 or full_h <= 0:
        raise ValueError("depth_pipeline full_w/full_h must be positive")

    command = [
        str(runtime["da3_python"]),
        "-u",
        str(STAGES / "run_da3_fixed_rig.py"),
        "--frames-dir",
        str(Path(config["frames_dir"]).resolve()),
        "--output-dir",
        str(Path(config["da3_self_cond_dir"]).resolve()),
        "--views",
        *[str(view) for view in config["views"]],
        "--timestamps",
        *timestamps,
        "--batch-size",
        str(int(runtime.get("batch_size", 1))),
        "--device",
        str(runtime.get("device", "cuda")),
        "--process-res",
        str(process_res),
        "--process-res-method",
        str(runtime.get("process_res_method", "upper_bound_resize")),
        "--ref-view-strategy",
        str(runtime.get("ref_view_strategy", "saddle_balanced")),
        "--full-w",
        str(full_w),
        "--full-h",
        str(full_h),
    ]
    camera_npz = runtime.get("camera_npz")
    camera_frames = runtime.get("camera_frames")
    if camera_npz:
        command.extend([
            "--camera-npz",
            str(Path(camera_npz).resolve()),
        ])
    elif camera_frames:
        command.extend([
            "--camera-frames",
            *[f"{int(frame):06d}" for frame in camera_frames],
        ])
    else:
        command.extend([
            "--camera-frame",
            f"{int(runtime['camera_frame']):06d}",
        ])
    if runtime.get("camera_frames_dir"):
        command.extend([
            "--camera-frames-dir",
            str(Path(runtime["camera_frames_dir"]).resolve()),
        ])
    if runtime.get("model_dir"):
        command.extend(["--model-dir", str(Path(runtime["model_dir"]).resolve())])
    if runtime.get("use_ray_pose", False):
        command.append("--use-ray-pose")
    if runtime.get("allow_legacy_shape_resume", False):
        command.append("--allow-legacy-shape-resume")
    if force:
        command.append("--overwrite")
    return command


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


def validate_reused_da3(
    output: Path,
    timestamps: Sequence[str],
) -> None:
    """Fail closed unless every requested reusable DA3 frame is present."""

    if len(timestamps) > 100_000:
        raise ValueError("refusing to validate more than 100000 DA3 frames")
    missing = [
        str(output / timestamp / "predictions.npz")
        for timestamp in timestamps
        if not (output / timestamp / "predictions.npz").is_file()
    ]
    if missing:
        preview = missing[:10]
        suffix = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise FileNotFoundError(
            "reusable DA3 artifact is incomplete: "
            + ", ".join(preview)
            + suffix
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=("all", "da3", "gauge", "cloud", "quality", "postprocess"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config.get("depth_pipeline", {})
    frames = config.get("frames", {})
    start = int(frames.get("start", 0))
    end = int(frames.get("end", 10**9))
    timestamps = [f"{frame:06d}" for frame in range(start, end + 1)]

    if args.stage in {"all", "da3"}:
        da3_output = Path(config["da3_self_cond_dir"]).resolve()
        if runtime.get("reuse_existing_da3", False):
            validate_reused_da3(da3_output, timestamps)
            print(
                f"[reuse] validated {len(timestamps)} DA3 frames at "
                f"{da3_output}",
                flush=True,
            )
        else:
            expected = da3_output / timestamps[-1] / "predictions.npz"
            command = build_da3_command(
                config, runtime, timestamps, force=args.force
            )
            _run(
                command,
                expected,
                args.force,
                stat_paths=(Path(config["frames_dir"]).resolve(),),
            )

    gauge_value = config.get("depth_gauge_path")
    if args.stage == "gauge" and not gauge_value:
        raise ValueError(
            "depth gauge stage requested but depth_gauge_path is disabled"
        )
    if args.stage in {"all", "gauge", "postprocess"} and gauge_value:
        gauge_path = Path(gauge_value).resolve()
        if runtime.get("reuse_existing_depth_gauge", False):
            if not gauge_path.is_file():
                raise FileNotFoundError(
                    f"configured reusable depth gauge is missing: {gauge_path}"
                )
            print(f"[reuse] depth gauge {gauge_path}", flush=True)
        else:
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

    quality_enabled = bool(config.get("quality_cloud", {}).get("enabled", False))
    if args.stage == "cloud" or (
        args.stage in {"all", "postprocess"} and not quality_enabled
    ):
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
                *_optional_path(config.get("depth_gauge_path")),
            ),
            stat_paths=(
                Path(config["masks_dir"]).resolve(),
                Path(config["da3_self_cond_dir"]).resolve(),
            ),
        )

    if args.stage == "quality" or (
        args.stage in {"all", "postprocess"} and quality_enabled
    ):
        quality = config.get("quality_cloud", {})
        quality_root = Path(
            quality.get(
                "point_cloud_root",
                Path(config.get(
                    "point_cloud_output_root", config["output_root"]
                ))
                / "parts_ply"
                / quality.get(
                    "variant", f"{config['recon_backend']}_quality"
                ),
            )
        ).resolve()
        expected = quality_root / "complete.json"
        _run(
            [
                sys.executable,
                "-u",
                str(STAGES / "build_quality_point_clouds.py"),
                "--pipeline",
                str(config_path),
            ],
            expected,
            args.force,
            content_files=(
                config_path,
                *_optional_path(config.get("depth_gauge_path")),
            ),
            stat_paths=(
                Path(config["masks_dir"]).resolve(),
                Path(config["da3_self_cond_dir"]).resolve(),
            ),
        )
