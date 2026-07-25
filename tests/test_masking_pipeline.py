from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common.masking.compose import compose_frame, compose_track_tree
from common.masking.io import load_label_mask, save_binary_mask, track_path
from common.masking.multiview import project_mask_to_view
from common.masking.quality import summarize_area_series
from common.masking.schema import load_mask_pipeline_config
from scripts.run_mask_pipeline import _sam_jobs, _seed_frames
from tools.stages.masking.track_part_masks import _seed_for


class MaskSchemaTests(unittest.TestCase):
    def write_config(self, root: Path, data: dict) -> Path:
        path = root / "mask.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_arbitrary_parts_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "work_root": str(root / "work"),
                "output_root": str(root / "output"),
                "views": ["left", "right"],
                "parts": {
                    "base": {
                        "id": 7,
                        "color": [1, 2, 3],
                        "start_frame": 12,
                        "prompts": ["object base"],
                    },
                    "handle": {
                        "id": 9,
                        "color": [4, 5, 6],
                        "start_frame": 20,
                        "prompts": ["handle"],
                    },
                },
                "occlusion_order": ["handle", "base"],
            }))
            self.assertEqual(config.part_names, ["base", "handle"])
            self.assertEqual(config.part_map["base"].id, 7)
            self.assertEqual(config.part_map["handle"].start_frame, 20)
            self.assertEqual(config.occlusion_order, ("handle", "base"))

    def test_duplicate_part_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": {
                    "one": {"id": 2},
                    "two": {"id": 2},
                },
                "occlusion_order": ["one", "two"],
            })
            with self.assertRaises(ValueError):
                load_mask_pipeline_config(path)

    def test_automatic_ids_do_not_collide_with_legacy_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": {
                    "custom": {},
                    "lid": {},
                    "body": {},
                },
                "occlusion_order": ["custom", "lid", "body"],
            }))
            ids = [part.id for part in config.parts]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(config.part_map["lid"].id, 1)
            self.assertEqual(config.part_map["body"].id, 2)
            self.assertEqual(config.part_map["custom"].id, 3)

    def test_per_view_seeds_and_segment_view_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["left", "right"],
                "parts": {
                    "body": {
                        "start_frame": 4,
                        "tracking": {
                            "mode": "fixed-image",
                            "seed_frames": {"default": 10, "right": 20},
                        },
                    },
                    "lid": {
                        "start_frame": 8,
                        "tracking": {
                            "mode": "video",
                            "seed_frame": 12,
                            "segments": [{
                                "views": ["right"],
                                "range": [8, 9],
                                "seed_frame": 9,
                            }],
                        },
                    },
                },
                "occlusion_order": ["lid", "body"],
            }))
            self.assertEqual(_seed_for(config, "body", "left", None), "000010")
            self.assertEqual(_seed_for(config, "body", "right", None), "000020")
            self.assertEqual(
                _seed_frames(config), ["000009", "000010", "000012", "000020"]
            )
            jobs = _sam_jobs(
                config,
                ["000000", "000001", "000002", "000003", "000004",
                 "000005", "000006", "000007", "000008", "000009",
                 "000010", "000011", "000012"],
                ["lid"],
                ["left"],
            )
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["views"], ["left"])


class MaskCompositionTests(unittest.TestCase):
    def config(self, root: Path):
        path = root / "mask.json"
        path.write_text(json.dumps({
            "frames_dir": str(root / "frames"),
            "views": ["cam"],
            "parts": {
                "back": {"id": 4, "start_frame": 5, "color": [0, 255, 0]},
                "front": {"id": 8, "start_frame": 10, "color": [255, 0, 0]},
            },
            "occlusion_order": ["front", "back"],
        }), encoding="utf-8")
        return load_mask_pipeline_config(path)

    def test_start_frames_and_front_to_back_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            back = np.zeros((5, 5), bool)
            front = np.zeros((5, 5), bool)
            back[1:4, 1:4] = True
            front[2:5, 2:5] = True
            label, resolved = compose_frame(
                {"back": back, "front": front}, config, 7, (5, 5)
            )
            self.assertFalse(resolved["front"].any())
            self.assertEqual(int((label == 4).sum()), 9)
            label, resolved = compose_frame(
                {"back": back, "front": front}, config, 10, (5, 5)
            )
            self.assertEqual(int((label == 8).sum()), 9)
            self.assertEqual(int((label == 4).sum()), 5)
            self.assertFalse((resolved["front"] & resolved["back"]).any())

    def test_quality_report_suggests_middle_of_empty_run(self):
        report = summarize_area_series(
            ["000004", "000005", "000006", "000007", "000008"],
            [0, 100, 0, 0, 100],
            start_frame=5,
        )
        self.assertEqual(report["empty_runs"], [[6, 7]])
        self.assertEqual(report["suggested_reanchor_frames"], [7])

    def test_track_tree_enforces_active_track_completeness(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            frame_root = root / "frames" / "cam"
            frame_root.mkdir(parents=True)
            for timestamp in ("000004", "000005", "000006"):
                Image.new("RGB", (8, 6)).save(frame_root / f"{timestamp}.jpg")
            back = np.zeros((6, 8), bool)
            back[1:4, 2:6] = True
            save_binary_mask(
                track_path(config.tracks_root, "back", "000005", "cam"),
                back,
            )
            # Missing tracks are allowed before start_frame.
            compose_track_tree(config, ["000004", "000005"])
            label = load_label_mask(config.masks_root / "000005" / "cam.png")
            self.assertEqual(int((label == 4).sum()), int(back.sum()))
            # Once a part exists, a missing file is an incomplete run, not
            # evidence that the object is invisible.
            with self.assertRaises(FileNotFoundError):
                compose_track_tree(config, ["000006"])


class MultiViewMaskTests(unittest.TestCase):
    def test_identity_cameras_preserve_mask(self):
        depth = np.full((20, 20), 2.0, np.float32)
        mask = np.zeros((20, 20), bool)
        mask[6:14, 7:13] = True
        intrinsic = np.array([
            [20.0, 0.0, 10.0],
            [0.0, 20.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        projected = project_mask_to_view(
            mask,
            depth,
            intrinsic,
            extrinsic,
            depth,
            intrinsic,
            extrinsic,
            depth_tolerance=1e-4,
            dilate=0,
            close_kernel=1,
        )
        np.testing.assert_array_equal(projected, mask)

    def test_target_occluder_rejects_projected_surface(self):
        source_depth = np.full((20, 20), 2.0, np.float32)
        target_depth = np.full((20, 20), 1.0, np.float32)
        mask = np.zeros((20, 20), bool)
        mask[6:14, 7:13] = True
        intrinsic = np.array([
            [20.0, 0.0, 10.0],
            [0.0, 20.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        projected = project_mask_to_view(
            mask,
            source_depth,
            intrinsic,
            extrinsic,
            target_depth,
            intrinsic,
            extrinsic,
            depth_tolerance=0.05,
            dilate=0,
            close_kernel=1,
        )
        self.assertFalse(projected.any())


if __name__ == "__main__":
    unittest.main()
