from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from common.io_utils import write_json
from pose_solver.artifacts import ArtifactLayout
from pose_solver.config import load_pipeline_config
from pose_solver.pipeline import PipelineRunner, inspect_result
from pose_solver.resolved import pose_config


class UnifiedPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frames = self.root / "frames"
        for view in ("cam0", "cam1"):
            (self.frames / view).mkdir(parents=True)
        self.meshes = self.root / "meshes"
        self.meshes.mkdir()
        (self.meshes / "piece.glb").write_bytes(b"glb")
        self.masks = self.root / "existing_masks"
        self.masks.mkdir()
        self.clouds = self.root / "existing_clouds"
        self.clouds.mkdir()
        self.trajectory = self.root / "trajectory.json"
        write_json(self.trajectory, {
            "parts": ["piece"],
            "frames": {"000002": {}},
            "reference_part": "piece",
        })

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _raw(self) -> dict:
        return {
            "schema_version": 1,
            "dataset": "fixture",
            "input": {
                "frames_dir": str(self.frames),
                "views": ["cam0", "cam1"],
                "frame_range": [2, 4],
            },
            "parts": {
                "piece": {
                    "id": 3,
                    "mesh": str(self.meshes / "piece.glb"),
                    "prompt": "test piece",
                    "appearance_hint": 2,
                    "reference": True,
                }
            },
            "output": {"root": str(self.root / "output")},
            "runtime": {"devices": [6, 7]},
            "models": {},
            "mask": {"mode": "reuse", "artifact": str(self.masks)},
            "depth": {"mode": "reuse", "artifact": str(self.clouds)},
            "pose": {"mode": "reuse", "artifact": str(self.trajectory)},
        }

    def _config(self, raw: dict | None = None) -> Path:
        path = self.root / "pipeline.json"
        write_json(path, raw or self._raw())
        return path

    def test_preflight_reuses_artifacts_without_stage_processes(self) -> None:
        config = load_pipeline_config(self._config())
        manifest = PipelineRunner(config).run("preflight")
        self.assertTrue(manifest.is_file())
        payload = json.loads(manifest.read_text())
        self.assertEqual(payload["devices"], [6, 7])
        self.assertEqual(payload["stages"]["preflight"]["status"], "complete")

    def test_inspect_is_read_only_and_summarizes_trajectory(self) -> None:
        config = load_pipeline_config(self._config())
        summary = inspect_result(config)
        self.assertEqual(summary["trajectory"]["parts"], ["piece"])
        self.assertEqual(summary["trajectory"]["frames"], 1)
        self.assertFalse(ArtifactLayout(config.output_root).manifest.exists())

    def test_rejects_more_than_two_devices(self) -> None:
        raw = self._raw()
        raw["runtime"]["devices"] = [5, 6, 7]
        with self.assertRaisesRegex(ValueError, "one or two unique GPUs"):
            load_pipeline_config(self._config(raw))

    def test_video_input_materializes_frame_stage_in_dry_run(self) -> None:
        raw = self._raw()
        destination = self.root / "extracted" / "frames"
        videos = {}
        for view in ("cam0", "cam1"):
            video = self.root / f"{view}.mp4"
            video.write_bytes(b"placeholder")
            videos[view] = str(video)
        raw["input"]["frames_dir"] = str(destination)
        raw["input"]["videos"] = videos
        config = load_pipeline_config(self._config(raw))
        PipelineRunner(config, dry_run=True).run("frames")
        resolved = ArtifactLayout(config.output_root).resolved_config("frames")
        self.assertTrue(resolved.is_file())
        payload = json.loads(resolved.read_text())
        self.assertEqual(set(payload["videos"]), {"cam0", "cam1"})

    def test_pose_compatibility_cannot_change_data_contract(self) -> None:
        legacy = self.root / "legacy_pose.json"
        write_json(legacy, {
            "views": ["wrong"],
            "parts": ["piece"],
            "states": {"piece": {"method": "cloud_registration"}},
            "registration": {
                "voxel_sizes_m": [0.01],
                "max_correspondence_m": [0.02],
                "max_iterations": 99,
            },
        })
        raw = self._raw()
        raw["pose"] = {
            "mode": "run",
            "compatibility_config": str(legacy),
            "overrides": {"registration": {"max_iterations": 12}},
        }
        config = load_pipeline_config(self._config(raw))
        resolved = pose_config(
            config,
            ArtifactLayout(config.output_root),
            {"piece": 2},
        )
        self.assertEqual(resolved["views"], ["cam0", "cam1"])
        self.assertEqual(resolved["output_root"], str(config.output_root / "pose"))
        self.assertEqual(resolved["states"]["piece"]["method"], "cloud_registration")
        self.assertEqual(resolved["registration"]["max_iterations"], 12)


if __name__ == "__main__":
    unittest.main()
