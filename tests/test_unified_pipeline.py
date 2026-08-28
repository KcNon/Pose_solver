from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from common.io_utils import write_json
from pose_solver.artifacts import ArtifactLayout
from pose_solver.cli import memory_guard_command
from pose_solver.config import load_pipeline_config
from pose_solver.pipeline import PipelineRunner, inspect_result
from pose_solver.resolved import depth_config, pose_config
from scripts.run_pose_pipeline import _resolve_render_views


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

    def test_render_stage_reuses_existing_trajectory(self) -> None:
        raw = self._raw()
        raw["pose"]["overrides"] = {
            "render": {"views": "all", "mask_occluder_labels": [3]},
        }
        config = load_pipeline_config(self._config(raw))

        PipelineRunner(config, dry_run=True).run("render")

        resolved = ArtifactLayout(config.output_root).resolved_config("pose")
        payload = json.loads(resolved.read_text())
        self.assertEqual(payload["render"]["views"], "all")
        self.assertEqual(payload["render"]["mask_occluder_labels"], [3])

    def test_render_view_resolution_is_centralized_and_validated(self) -> None:
        config = {
            "views": ["cam0", "cam1"],
            "render": {"views": "all", "primary_view": "cam1"},
        }
        self.assertEqual(_resolve_render_views(config), ["cam0", "cam1"])
        self.assertEqual(_resolve_render_views(config, "cam1"), ["cam1"])
        config["render"]["views"] = ["cam0", "cam0"]
        with self.assertRaisesRegex(ValueError, "unique"):
            _resolve_render_views(config)
        with self.assertRaisesRegex(ValueError, "unknown render views"):
            _resolve_render_views(config, "cam2")

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

    def test_explicit_egl_device_is_preserved(self) -> None:
        raw = self._raw()
        raw["runtime"]["egl_device"] = 15
        config = load_pipeline_config(self._config(raw))
        self.assertEqual(PipelineRunner(config)._environment()["EGL_DEVICE_ID"], "15")

    def test_memory_guard_is_enabled_by_default(self) -> None:
        config = load_pipeline_config(self._config())
        self.assertTrue(config.memory_guard.enabled)
        self.assertEqual(config.memory_guard.minimum_available_gib, 128.0)
        self.assertEqual(config.memory_guard.maximum_process_rss_gib, 32.0)
        command = memory_guard_command(
            config, ["run", "--config", str(config.source_path)]
        )
        self.assertIn("run_with_memory_guard.py", " ".join(command))
        self.assertIn("6,7", command)
        self.assertEqual(PipelineRunner(config)._environment()["EGL_DEVICE_ID"], "6")

    def test_rejects_invalid_memory_guard_limits(self) -> None:
        raw = self._raw()
        raw["runtime"]["memory_guard"] = {
            "minimum_available_gib": 0,
        }
        with self.assertRaisesRegex(ValueError, "must be positive"):
            load_pipeline_config(self._config(raw))

    def test_rejects_disabled_memory_guard(self) -> None:
        raw = self._raw()
        raw["runtime"]["memory_guard"] = {"enabled": False}
        with self.assertRaisesRegex(ValueError, "cannot be disabled"):
            load_pipeline_config(self._config(raw))

    def test_mesh_directory_contract_preserves_symlink_collection(self) -> None:
        targets = self.root / "reconstructed_parts"
        for part in ("body", "lid"):
            target = targets / part / "model.glb"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"glb")
            (self.meshes / f"{part}.glb").symlink_to(target)
        raw = self._raw()
        raw["parts"] = {
            "body": {
                "id": 1,
                "mesh": str(self.meshes / "body.glb"),
                "prompt": "body",
                "appearance_hint": 2,
                "reference": True,
            },
            "lid": {
                "id": 2,
                "mesh": str(self.meshes / "lid.glb"),
                "prompt": "lid",
                "appearance_hint": 2,
            },
        }

        config = load_pipeline_config(self._config(raw))

        self.assertEqual(config.mesh_dir, self.meshes)
        self.assertNotEqual(
            config.parts[0].mesh.resolve().parent,
            config.parts[1].mesh.resolve().parent,
        )

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

    def test_view_first_reused_masks_are_normalized_for_pose_stages(self) -> None:
        raw = self._raw()
        view_first = self.root / "view_first_masks"
        for view in ("cam0", "cam1"):
            directory = view_first / view
            directory.mkdir(parents=True)
            for frame in range(2, 5):
                (directory / f"{frame:06d}.png").write_bytes(b"mask")
        raw["mask"] = {
            "mode": "reuse",
            "artifact": str(view_first),
            "overrides": {"source_layout": "view_first"},
        }
        config = load_pipeline_config(self._config(raw))

        PipelineRunner(config).run("pose")

        normalized = config.output_root / "mask" / "masks"
        destination = normalized / "000003" / "cam1.png"
        self.assertTrue(destination.is_symlink())
        self.assertEqual(
            destination.resolve(),
            (view_first / "cam1" / "000003.png").resolve(),
        )

    def test_invalid_reused_mask_layout_is_rejected(self) -> None:
        raw = self._raw()
        raw["mask"]["overrides"] = {"source_layout": "camera_major"}
        with self.assertRaisesRegex(ValueError, "source_layout"):
            load_pipeline_config(self._config(raw))

    def test_existing_da3_input_is_postprocessed_without_model_launch(self) -> None:
        raw = self._raw()
        raw["input"]["depth_dir"] = str(self.root / "existing_da3")
        raw["depth"] = {"mode": "run"}
        config = load_pipeline_config(self._config(raw))

        resolved = depth_config(
            config,
            ArtifactLayout(config.output_root),
            {"piece": 2},
        )

        self.assertTrue(resolved["depth_pipeline"]["reuse_existing_da3"])
        self.assertEqual(resolved["depth_pipeline"]["da3_python"], "")

    def test_pose_config_routes_generated_da3_artifact(self) -> None:
        raw = self._raw()
        raw["depth"] = {"mode": "run"}
        raw["pose"] = {
            "mode": "run",
            "overrides": {"da3_self_cond_dir": "/wrong/depth"},
        }
        config = load_pipeline_config(self._config(raw))
        layout = ArtifactLayout(config.output_root)

        resolved = pose_config(config, layout, {"piece": 2})

        self.assertEqual(
            resolved["da3_self_cond_dir"],
            str(layout.depth_output / "da3-self-cond"),
        )


if __name__ == "__main__":
    unittest.main()
