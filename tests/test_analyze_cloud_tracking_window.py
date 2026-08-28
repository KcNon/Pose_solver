import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tools.diagnostics.analyze_cloud_tracking_window import analyze_window


def pose(x: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    value[0, 3] = x
    return value


class AnalyzeCloudTrackingWindowTests(unittest.TestCase):
    def test_separates_raw_path_from_endpoint_correction(self):
        trajectory = {"frames": {}}
        for frame, value in ((10, pose()), (11, pose(0.06, 5.0)), (12, pose(0.12, 10.0))):
            trajectory["frames"][f"{frame:06d}"] = {
                "parts": {"pump": {"T_world_from_part": value.tolist()}}
            }
        pair = pose(-0.01, -1.0)
        registrations = {
            "pump": {
                f"{frame:06d}_to_{frame - 1:06d}": {
                    "T_target_from_source": pair.tolist(),
                    "quality": {
                        "translation_m": 0.01,
                        "rotation_deg": 1.0,
                        "fitness_8mm": 0.9,
                        "median_nn_m": 0.001,
                        "rejected": False,
                    },
                }
                for frame in (11, 12)
            }
        }

        report = analyze_window(
            trajectory, registrations, part="pump", start=10, end=12
        )

        self.assertAlmostEqual(
            report["maxima"]["raw_translation_step"]["translation_m"],
            0.01,
            places=6,
        )
        self.assertGreater(
            report["raw_predicted_end_to_forced_end_pose_mismatch"]["translation_m"],
            0.09,
        )
        self.assertGreater(
            report["maxima"]["saved_translation_step"]["translation_m"], 0.05
        )
        self.assertLess(
            report["maxima"]["origin_safe_translation_step"]["translation_m"],
            0.07,
        )


if __name__ == "__main__":
    unittest.main()
