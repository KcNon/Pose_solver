from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common.simulation_assets import (
    canonical_from_raw_matrix,
    create_collision_proxy,
    robust_average_pose,
    rotation_align_vectors,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/simulation_data_1.json").read_text())
ASSET_ROOT = ROOT / CONFIG["output_dir"]


class SimulationAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = json.loads(
            (ROOT / CONFIG["trajectory"]).read_text()
        )
        manifest_path = ASSET_ROOT / "manifest.json"
        cls.manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )

    def test_raw_to_world_transform_matches_solver(self) -> None:
        for part in self.trajectory["parts"]:
            canonical = canonical_from_raw_matrix(
                self.trajectory["scales"][part], self.trajectory["raw_mesh_origins"][part]
            )
            for frame in self.trajectory["frames"].values():
                data = frame["parts"][part]
                pose = np.asarray(data["T_world_from_part"])
                render = np.asarray(data["S_world_from_raw_mesh"])
                np.testing.assert_allclose(pose @ canonical, render, atol=1e-10)

    def test_average_pose_rejects_outlier(self) -> None:
        poses = [np.eye(4) for _ in range(5)]
        poses[-1] = np.eye(4)
        poses[-1][:3, 3] = [0.5, 0.0, 0.0]
        average, stats = robust_average_pose(poses, [str(index) for index in range(5)])
        np.testing.assert_allclose(average, np.eye(4), atol=1e-9)
        self.assertEqual(stats["rejected_frames"], ["4"])

    def test_up_axis_alignment(self) -> None:
        rotation = rotation_align_vectors([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_cylindrical_container_proxy_preserves_cavity(self) -> None:
        visual = __import__("trimesh").creation.box()
        proxy, info = create_collision_proxy(
            visual,
            {
                "type": "cylindrical_container",
                "axis": "y",
                "inner_radius_m": 0.12,
                "outer_radius_m": 0.15,
                "floor_top_m": -0.01,
                "rim_top_m": 0.11,
                "floor_thickness_m": 0.01,
                "sections": 32,
            },
        )
        self.assertTrue(proxy.is_watertight)
        self.assertEqual(proxy.body_count, 2)
        self.assertEqual(info["type"], "cylindrical_container")
        np.testing.assert_allclose(proxy.bounds[:, 1], [-0.02, 0.11], atol=1e-12)

    def test_compound_insert_proxy_is_closed_and_low_poly(self) -> None:
        visual = __import__("trimesh").creation.box()
        proxy, info = create_collision_proxy(
            visual,
            {
                "type": "compound",
                "components": [
                    {
                        "type": "revolved_solid",
                        "axis": "y",
                        "profile_axis_radius_m": [[-0.07, 0.09], [0.05, 0.115]],
                        "sections": 32,
                    },
                    {
                        "type": "cylinder",
                        "axis": "y",
                        "radius_m": 0.14,
                        "axis_min_m": 0.05,
                        "axis_max_m": 0.08,
                        "sections": 32,
                    },
                ],
            },
        )
        self.assertTrue(proxy.is_watertight)
        self.assertEqual(proxy.body_count, 2)
        self.assertLess(len(proxy.faces), 1000)
        self.assertEqual(len(info["components"]), 2)

    def test_uniform_scale_applies_to_configured_proxy(self) -> None:
        trimesh = __import__("trimesh")
        visual = trimesh.creation.box()
        spec = {
            "type": "cylinder",
            "axis": "z",
            "radius_m": 0.2,
            "axis_min_m": -0.1,
            "axis_max_m": 0.1,
            "sections": 32,
        }
        baseline, _ = create_collision_proxy(visual, spec)
        scaled, info = create_collision_proxy(
            visual, {**spec, "uniform_scale": 0.94}
        )
        np.testing.assert_allclose(
            scaled.extents,
            baseline.extents * 0.94,
            rtol=1e-7,
            atol=1e-9,
        )
        self.assertEqual(info["uniform_scale"], 0.94)

    def test_urdf_mesh_paths_exist(self) -> None:
        urdf_paths = sorted((ASSET_ROOT / "urdf").glob("*.urdf"))
        if not urdf_paths:
            self.skipTest("simulation assets have not been exported")
        for urdf_path in urdf_paths:
            root = ET.parse(urdf_path).getroot()
            self.assertEqual(root.tag, "robot")
            for mesh in root.findall(".//mesh"):
                resolved = (urdf_path.parent / mesh.attrib["filename"]).resolve()
                self.assertTrue(resolved.is_file(), f"Missing mesh referenced by {urdf_path}: {resolved}")

    def test_manifest_has_independent_and_display_assets(self) -> None:
        if self.manifest is None:
            self.skipTest("simulation assets have not been exported")
        outputs = self.manifest["outputs"]
        self.assertTrue((ASSET_ROOT / outputs["display_urdf"]).is_file())
        self.assertEqual(set(outputs["independent_urdfs"]), {"body", "inner_pot", "lid"})
        transform = np.asarray(self.manifest["assembled_T_body_from_part"]["inner_pot"])
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)

    def test_display_urdf_fixed_joints_match_manifest(self) -> None:
        if self.manifest is None:
            self.skipTest("simulation assets have not been exported")
        root = ET.parse(ASSET_ROOT / self.manifest["outputs"]["display_urdf"]).getroot()
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
            rpy = np.fromstring(origin.attrib["rpy"], sep=" ")
            actual = np.eye(4)
            actual[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            actual[:3, 3] = xyz
            expected = np.asarray(self.manifest["assembled_T_body_from_part"][child])
            np.testing.assert_allclose(actual, expected, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
