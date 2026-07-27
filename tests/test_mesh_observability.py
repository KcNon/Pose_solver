from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import trimesh

from common.mesh_observability import infer_mesh_observability


class MeshObservabilityTests(unittest.TestCase):
    def test_cylinder_proposes_axial_symmetry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cylinder.glb"
            trimesh.creation.cylinder(radius=1.0, height=1.5, sections=96).export(
                path
            )
            report = infer_mesh_observability(path)
            self.assertEqual(
                report["symmetry"]["equivalence"],
                "continuous_axial",
            )

    def test_asymmetric_box_does_not_propose_continuous_symmetry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "box.glb"
            mesh = trimesh.creation.box(extents=[1.0, 2.0, 3.0])
            bump = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
            bump.apply_translation([0.55, 0.6, 0.8])
            trimesh.util.concatenate((mesh, bump)).export(path)
            report = infer_mesh_observability(path)
            self.assertNotEqual(
                report["symmetry"]["equivalence"],
                "continuous_axial",
            )


if __name__ == "__main__":
    unittest.main()
