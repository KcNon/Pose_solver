from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common.simulation_assets import (
    canonical_from_raw_matrix,
    carve_observed_overlap_components,
    create_collision_proxy,
    robust_average_pose,
    rotation_align_vectors,
)
from common.simulation_autoconfig import final_state_run
from common.simulation_export import materialize_collision_components


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "experiments/data_1/simulation_assets_scale_calibrated"


def _synthetic_trajectory() -> dict:
    pose = np.eye(4)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    canonical = np.diag([2.0, 2.0, 2.0, 1.0])
    canonical[:3, 3] = [-2.0, -4.0, -6.0]
    return {
        "parts": ["sample"],
        "scales": {"sample": 2.0},
        "raw_mesh_origins": {"sample": [1.0, 2.0, 3.0]},
        "frames": {
            "000000": {
                "parts": {
                    "sample": {
                        "T_world_from_part": pose.tolist(),
                        "S_world_from_raw_mesh": (pose @ canonical).tolist(),
                    }
                }
            }
        },
    }


class SimulationAssetTests(unittest.TestCase):
    def test_dense_proxy_component_split_fails_closed(self) -> None:
        mesh = __import__("trimesh").creation.box()
        self.assertEqual(
            materialize_collision_components(mesh, proxy_type="raw"),
            [mesh],
        )
        with self.assertRaisesRegex(RuntimeError, "refusing to materialize"):
            materialize_collision_components(
                mesh, proxy_type="filtered_surface", maximum_faces=1
            )

    def test_final_state_run_chooses_last_contiguous_visible_run(self) -> None:
        rows = {
            "000010": {"state": "assembled", "observing_views": 2},
            "000011": {"state": "assembled", "observing_views": 0},
            "000020": {"state": "assembled", "observing_views": 3},
            "000021": {"state": "assembled", "observing_views": 2},
        }
        self.assertEqual(final_state_run(rows, {"assembled"}), (20, 21))

    @classmethod
    def setUpClass(cls) -> None:
        cls.trajectory = _synthetic_trajectory()
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

    def test_oriented_cylinder_follows_connector_axis(self) -> None:
        visual = __import__("trimesh").creation.box()
        axis = np.asarray([1.0, 2.0, -1.0], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        origin = np.asarray([0.03, -0.02, 0.01], dtype=np.float64)
        proxy, info = create_collision_proxy(
            visual,
            {
                "type": "oriented_cylinder",
                "axis_vector": axis.tolist(),
                "origin_m": origin.tolist(),
                "radius_m": 0.01,
                "axis_min_m": -0.02,
                "axis_max_m": 0.04,
                "sections": 32,
            },
        )
        self.assertTrue(proxy.is_watertight)
        self.assertEqual(info["type"], "oriented_cylinder")
        np.testing.assert_allclose(
            proxy.centroid,
            origin + 0.01 * axis,
            atol=1e-9,
        )

    def test_annular_cylinder_preserves_center_passage(self) -> None:
        visual = __import__("trimesh").creation.box()
        proxy, info = create_collision_proxy(
            visual,
            {
                "type": "annular_cylinder",
                "axis": "y",
                "inner_radius_m": 0.01,
                "outer_radius_m": 0.02,
                "axis_min_m": 0.1,
                "axis_max_m": 0.104,
                "sections": 32,
            },
        )
        self.assertTrue(proxy.is_watertight)
        self.assertEqual(info["type"], "annular_cylinder")
        np.testing.assert_allclose(proxy.bounds[:, 1], [0.1, 0.104], atol=1e-12)

    def test_oriented_annular_cylinder_is_convex_component_sleeve(self) -> None:
        visual = __import__("trimesh").creation.box()
        axis = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        origin = np.asarray([0.02, 0.03, -0.01], dtype=np.float64)
        proxy, info = create_collision_proxy(
            visual,
            {
                "type": "cylindrical_sleeve",
                "axis_vector": axis.tolist(),
                "origin_m": origin.tolist(),
                "inner_radius_m": 0.0165,
                "outer_radius_m": 0.020,
                "axis_min_m": -0.004,
                "axis_max_m": 0.012,
                "sections": 16,
                "parameter_source": "measured",
            },
        )
        self.assertTrue(proxy.is_watertight)
        self.assertEqual(proxy.body_count, 16)
        self.assertEqual(info["type"], "cylindrical_sleeve")
        self.assertEqual(info["convex_components"], 16)
        self.assertTrue(info["center_passage_preserved"])
        self.assertEqual(info["parameter_source"], "measured")
        np.testing.assert_allclose(
            proxy.centroid,
            origin + 0.004 * axis,
            atol=2e-5,
        )

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

    def test_voxel_shell_is_watertight_compound_without_filling_cavity(self) -> None:
        trimesh = __import__("trimesh")
        outer = trimesh.creation.box(extents=[0.12, 0.12, 0.12])
        # Remove the top and bottom faces so the input is intentionally
        # non-watertight and has an observed passage through its center.
        normals = outer.face_normals
        shell = outer.submesh(
            [np.flatnonzero(np.abs(normals[:, 2]) < 0.9)],
            append=True,
            repair=False,
        )
        proxy, info = create_collision_proxy(
            shell,
            {"type": "voxel_shell", "pitch_m": 0.02, "resolution": 12},
        )
        self.assertTrue(proxy.is_watertight)
        self.assertGreater(info["convex_components"], 1)
        self.assertLess(info["convex_components"], info["surface_voxels"])
        centers = np.asarray(
            [component.centroid for component in proxy.split(only_watertight=False)]
        )
        self.assertFalse(np.any(np.linalg.norm(centers, axis=1) < 0.01))

    def test_filtered_surface_removes_small_reconstruction_fragments(self) -> None:
        trimesh = __import__("trimesh")
        main = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        fragment = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
        fragment.apply_translation([3.0, 0.0, 0.0])
        visual = trimesh.util.concatenate([main, fragment])
        proxy, info = create_collision_proxy(
            visual,
            {"type": "filtered_surface", "minimum_component_faces": 100},
        )
        self.assertEqual(info["input_components"], 2)
        self.assertEqual(info["kept_components"], 1)
        self.assertEqual(proxy.body_count, 1)
        self.assertLess(proxy.bounds[1, 0], 2.0)

    def test_observed_overlap_carves_crossed_cell_not_boundary_contact(self) -> None:
        trimesh = __import__("trimesh")
        first = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        second = trimesh.creation.box(
            extents=[0.1, 0.1, 0.1],
            transform=trimesh.transformations.translation_matrix([0.2, 0.0, 0.0]),
        )
        reference = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [-0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, -0.01, 0.0],
                [0.15, 0.0, 0.0],  # boundary contact on the second box
            ]
        )
        kept, report = carve_observed_overlap_components(
            [first, second],
            reference,
            np.eye(4),
            penetration_tolerance_m=0.001,
            minimum_reference_vertices=3,
            maximum_removed_fraction=0.6,
        )
        self.assertTrue(report["applied"])
        self.assertEqual(report["removed_components"], 1)
        self.assertEqual(len(kept), 1)
        np.testing.assert_allclose(
            kept[0].centroid, [0.2, 0.0, 0.0], atol=1e-15
        )

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
