from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common.io_utils import load_json, write_json
from common.mask_io import list_timestamps, view_names
from common.pose_transforms import (
    axis_rotation,
    axis_rotation_degrees,
    decompose_similarity,
    rigid_from_similarity,
    similarity,
    similarity_from_rigid,
    transform_points,
)


class JsonIoTests(unittest.TestCase):
    def test_json_round_trip_creates_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "value.json"
            value = {"part": "inner_pot", "unicode": "位姿", "frames": [0, 1]}
            write_json(path, value)
            self.assertEqual(load_json(path), value)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))

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
