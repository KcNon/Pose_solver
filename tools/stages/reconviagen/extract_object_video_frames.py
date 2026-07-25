#!/usr/bin/env python
"""Extract independently sampled frames from each canonical-part video."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    fps = float(config.get("sample_fps", 2.0))
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": str(config_path),
        "sample_fps": fps,
        "parts": {},
    }
    for part, source_value in config["videos"].items():
        source = Path(source_value).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        frame_dir = output_root / part / "frames" / part
        frame_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(frame_dir.glob("*.jpg"))
        if existing and not args.force:
            raise FileExistsError(
                f"{frame_dir} already contains {len(existing)} frames; use --force"
            )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-vf", f"fps={fps:.12g}",
            "-q:v", "2",
            "-start_number", "0",
            str(frame_dir / "%06d.jpg"),
        ]
        print(f"[{part}] {source} -> {frame_dir}", flush=True)
        subprocess.run(command, check=True)
        frames = sorted(frame_dir.glob("*.jpg"))
        if not frames:
            raise RuntimeError(f"no frames extracted for {part}")
        manifest["parts"][part] = {
            "video": str(source),
            "frames_dir": str(frame_dir.parent),
            "view": part,
            "frame_count": len(frames),
            "first_timestamp": frames[0].stem,
            "last_timestamp": frames[-1].stem,
        }

    manifest_path = output_root / "frame_extraction.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
