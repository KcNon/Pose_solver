#!/usr/bin/env python
"""Extract a common timestamp grid from independently started camera videos."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    return float(subprocess.check_output(command, text=True).strip())


def image_ids(directory: Path) -> list[str]:
    return sorted(
        path.stem
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir = Path(config["frames_dir"]).resolve()
    fps = float(config["sample_fps"])
    if fps <= 0:
        raise ValueError("sample_fps must be positive")

    views = [str(view) for view in config["views"]]
    videos = {str(key): Path(value).resolve() for key, value in config["videos"].items()}
    offsets = {
        str(key): float(value)
        for key, value in config.get("sync_offsets_s", {}).items()
    }
    if set(videos) != set(views):
        raise ValueError("videos keys must exactly match views")
    unknown_offsets = set(offsets) - set(views)
    if unknown_offsets:
        raise ValueError(f"sync offsets contain unknown views: {sorted(unknown_offsets)}")

    reference_trim = float(config.get("reference_trim_s", 0.0))
    starts = {view: reference_trim + offsets.get(view, 0.0) for view in views}
    if min(starts.values()) < 0:
        raise ValueError(
            "reference_trim_s is too small for the most negative sync offset: "
            f"{min(starts.values()):.6f}s"
        )

    durations = {}
    for view in views:
        if not videos[view].is_file():
            raise FileNotFoundError(videos[view])
        durations[view] = probe_duration(videos[view])
    available = {view: durations[view] - starts[view] for view in views}
    duration = min(available.values())
    requested = config.get("duration_s")
    if requested is not None:
        duration = min(duration, float(requested))
    if duration <= 0:
        raise ValueError("no common video duration remains after synchronization")

    output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        view_dir = output_dir / view
        view_dir.mkdir(parents=True, exist_ok=True)
        existing = image_ids(view_dir)
        if existing and not args.force:
            raise FileExistsError(
                f"{view_dir} already contains {len(existing)} frames; "
                "use --force to overwrite the same frame names"
            )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(videos[view]),
            "-ss", f"{starts[view]:.6f}",
            "-t", f"{duration:.6f}",
            "-vf", f"fps={fps:.12g}",
            "-q:v", "2",
            "-start_number", "0",
            str(view_dir / "%06d.jpg"),
        ]
        print(
            f"[{view}] source_start={starts[view]:.3f}s "
            f"duration={duration:.3f}s fps={fps:g}",
            flush=True,
        )
        subprocess.run(command, check=True)

    per_view_ids = {view: image_ids(output_dir / view) for view in views}
    reference_ids = per_view_ids[views[0]]
    mismatched = [
        view for view in views[1:] if per_view_ids[view] != reference_ids
    ]
    if mismatched:
        raise RuntimeError(f"extracted timestamp grids differ for views: {mismatched}")
    if not reference_ids:
        raise RuntimeError("frame extraction produced no images")

    manifest = {
        "config": str(config_path),
        "frames_dir": str(output_dir),
        "views": views,
        "videos": {view: str(videos[view]) for view in views},
        "sync_offsets_s_relative_to_reference": {
            view: offsets.get(view, 0.0) for view in views
        },
        "source_start_s": starts,
        "source_duration_s": durations,
        "common_duration_s": duration,
        "sample_fps": fps,
        "frame_count": len(reference_ids),
        "first_timestamp": reference_ids[0],
        "last_timestamp": reference_ids[-1],
    }
    manifest_path = output_dir.parent / "frame_extraction.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"wrote {len(reference_ids)} synchronized frames/view -> {output_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
