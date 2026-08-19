from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from common.backproject_utils import load_palette_masks
from common.depth_gauge import gauge_masks, reference_part
from common.io_utils import load_json, write_json
from common.mask_io import list_timestamps, view_names
from common.normalized_recon import load_recon
from common.pose_transforms import (
    axis_rotation,
    axis_rotation_degrees,
    decompose_similarity,
    rigid_from_similarity,
    similarity,
    similarity_from_rigid,
    transform_points,
)
from common.pose_visualization import tile_image_panels
from common.stage_cache import (
    checkpoint_matches,
    stage_fingerprint,
    write_checkpoint,
)


class JsonIoTests(unittest.TestCase):
    def test_depth_reference_falls_back_to_first_configured_part(self):
        self.assertEqual(
            reference_part({"parts": ["static_base", "moving_piece"]}),
            "static_base",
        )
        self.assertEqual(
            reference_part(
                {"parts": ["static_base"], "reference_part": "anchor"}
            ),
            "anchor",
        )

    def test_background_gauge_excludes_every_part(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "000000"
            frame.mkdir(parents=True)
            labels = np.zeros((4, 5), np.uint8)
            labels[1, 1] = 1
            labels[2, 3] = 2
            Image.fromarray(labels).save(frame / "view.png")
            masks = gauge_masks(
                {
                    "masks_dir": str(root),
                    "parts": ["main", "collector"],
                    "part_ids": {"main": 1, "collector": 2},
                    "views": ["view"],
                },
                "000000",
                (4, 5),
                "__background__",
            )
            self.assertFalse(masks[0][1, 1])
            self.assertFalse(masks[0][2, 3])
            self.assertEqual(int(masks[0].sum()), 18)

    def test_json_round_trip_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.json"
            value = {"part": "inner_pot", "unicode": "位姿", "frames": [0, 1]}
            write_json(path, value)
            self.assertEqual(load_json(path), value)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

    def test_stage_checkpoint_invalidates_on_nested_input_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            nested = inputs / "value.txt"
            nested.write_text("first", encoding="utf-8")
            expected = root / "result.json"
            expected.write_text("{}", encoding="utf-8")
            first = stage_fingerprint(
                command=["tool", "--run"],
                stat_paths=[inputs],
            )
            write_checkpoint(expected, first)
            self.assertTrue(checkpoint_matches(expected, first))
            nested.write_text("second value", encoding="utf-8")
            second = stage_fingerprint(
                command=["tool", "--run"],
                stat_paths=[inputs],
            )
            self.assertFalse(checkpoint_matches(expected, second))

    def test_configured_eight_view_layout(self) -> None:
        views = [f"camera_{index}" for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory)
            for view in views:
                view_dir = frames / view
                view_dir.mkdir()
                (view_dir / "000000.jpg").touch()
                (view_dir / "000001.jpg").touch()
            self.assertEqual(view_names({"views": views}), views)
            self.assertEqual(
                list_timestamps(str(frames), "normalized", views),
                ["000000", "000001"],
            )

    def test_duplicate_view_names_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            view_names({"views": ["camera_a", "camera_a"]})

    def test_palette_mask_preserves_configured_label_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "000000"
            frame.mkdir()
            labels = np.asarray([[0, 1, 3], [3, 1, 0]], dtype=np.uint8)
            image = Image.fromarray(labels, mode="P")
            image.putpalette([0, 0, 0] * 256)
            image.save(frame / "camera.png")
            masks = load_palette_masks(
                str(root),
                "000000",
                ["lid"],
                (2, 3),
                views=["camera"],
                part_ids={"lid": 3},
            )
            np.testing.assert_array_equal(masks["lid"][0], labels == 3)

    def test_reconstruction_arrays_follow_configured_view_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "000000"
            frame.mkdir()
            np.savez(
                frame / "predictions.npz",
                depth=np.arange(24, dtype=np.float32).reshape(3, 2, 4, 1),
                depth_conf=np.arange(24, dtype=np.float32).reshape(3, 2, 4),
                intrinsic=np.stack([np.eye(3) * value for value in (1, 2, 3)]),
                extrinsic=np.stack([np.eye(4)[:3] * value for value in (1, 2, 3)]),
                images=np.stack([
                    np.full((3, 2, 4), value, np.float32)
                    for value in (10, 20, 30)
                ]),
                view_names=np.asarray(["left", "middle", "right"]),
            )
            recon = load_recon(
                {
                    "da3_self_cond_dir": str(root),
                    "views": ["right", "left"],
                },
                "000000",
            )
            self.assertEqual(recon["view_names"], ["right", "left"])
            self.assertEqual(recon["n_views"], 2)
            self.assertEqual(float(recon["depth"][0, 0, 0]), 16.0)
            self.assertEqual(float(recon["depth"][1, 0, 0]), 0.0)
            self.assertEqual(float(recon["images"][0, 0, 0, 0]), 30.0)

    def test_reconstruction_subset_requires_view_name_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "000000"
            frame.mkdir()
            np.savez(
                frame / "predictions.npz",
                depth=np.ones((3, 2, 2, 1), np.float32),
                depth_conf=np.ones((3, 2, 2), np.float32),
                intrinsic=np.stack([np.eye(3)] * 3),
                extrinsic=np.stack([np.eye(4)[:3]] * 3),
            )
            with self.assertRaisesRegex(ValueError, "cannot select configured subset"):
                load_recon(
                    {
                        "da3_self_cond_dir": str(root),
                        "views": ["left", "right"],
                    },
                    "000000",
                )

    def test_image_panel_tiling_pads_incomplete_final_row(self) -> None:
        panels = [
            np.full((2, 3, 3), value, np.uint8) for value in range(6)
        ]
        sheet = tile_image_panels(panels, columns=4)
        self.assertEqual(sheet.shape, (4, 12, 3))
        np.testing.assert_array_equal(sheet[:2, 9:12], panels[3])
        np.testing.assert_array_equal(sheet[2:, 3:6], panels[5])
        self.assertFalse(sheet[2:, 6:].any())


class PoseTransformTests(unittest.TestCase):
    def test_similarity_rigid_round_trip(self) -> None:
        scale = 0.37
        origin = np.array([0.2, -0.1, 0.3])
        rigid = np.eye(4)
        rigid[:3, :3] = Rotation.from_euler("xyz", [15.0, -20.0, 35.0], degrees=True).as_matrix()
        rigid[:3, 3] = [0.4, 0.2, -0.15]

        encoded = similarity_from_rigid(rigid, scale, origin)
        decoded = rigid_from_similarity(encoded, origin)
        recovered_scale, recovered_rotation, _ = decompose_similarity(encoded)

        np.testing.assert_allclose(decoded, rigid, atol=1e-12)
        self.assertAlmostEqual(recovered_scale, scale)
        np.testing.assert_allclose(recovered_rotation, rigid[:3, :3], atol=1e-12)

    def test_similarity_constructor_and_point_transform(self) -> None:
        rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
        transform = similarity(2.0, rotation, np.array([1.0, 2.0, 3.0]))
        actual = transform_points(np.array([[1.0, 0.0, 0.0]]), transform)
        np.testing.assert_allclose(actual, [[1.0, 4.0, 3.0]], atol=1e-12)

    def test_radian_and_degree_axis_rotations_agree(self) -> None:
        axis = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            axis_rotation(axis, np.pi / 3.0),
            axis_rotation_degrees(axis, 60.0),
            atol=1e-12,
        )

    def test_invalid_similarity_and_axis_are_rejected(self) -> None:
        reflected = np.eye(4)
        reflected[0, 0] = -1.0
        with self.assertRaises(ValueError):
            decompose_similarity(reflected)
        with self.assertRaises(ValueError):
            axis_rotation(np.zeros(3), 1.0)


if __name__ == "__main__":
    unittest.main()
