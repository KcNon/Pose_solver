from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.diagnostics.validate_simulation_manifest import build_preflight_report


class SimulationValidationPreflightTests(unittest.TestCase):
    def test_missing_interface_and_failed_connector_block_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("qa/geometry.json", "qa/replay.json", "urdf/a.urdf"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            manifest = {
                "outputs": {
                    "geometry_report": "qa/geometry.json",
                    "observed_replay": "qa/replay.json",
                    "independent_urdfs": {"a": "urdf/a.urdf"},
                }
            }
            report = build_preflight_report(
                manifest,
                {
                    "validation_readiness": {
                        "runnable": False,
                        "failures": ["assembly_interface_missing"],
                    }
                },
                manifest_path=root / "manifest.json",
                connector_evidence={
                    "connectors": {
                        "x": {
                            "simulation_ready": False,
                            "failures": ["axis_angle_exceeds_limit"],
                        }
                    }
                },
            )
            self.assertFalse(report["physics_launch_allowed"])
            self.assertIn("assembly_interface_missing", report["failures"])
            self.assertIn(
                "external_connector_evidence_not_simulation_ready",
                report["failures"],
            )


if __name__ == "__main__":
    unittest.main()
