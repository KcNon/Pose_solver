from __future__ import annotations

import copy
import unittest

import numpy as np

from tools.stages.pose.compose_trajectory_parts import compose_trajectory_parts


def _trajectory(x: float) -> dict:
    identity = np.eye(4)
    identity[0, 3] = x
    return {
        "parts": ["moving", "body"],
        "reference_part": "body",
        "scales": {"moving": 1.0, "body": 2.0 + x},
        "raw_mesh_origins": {"moving": [0, 0, 0], "body": [0, 0, 0]},
        "frames": {
            "000001": {"parts": {
                "moving": {"T_world_from_part": np.eye(4).tolist()},
                "body": {"T_world_from_part": identity.tolist()},
            }}
        },
    }


class ComposeTrajectoryPartsTests(unittest.TestCase):
    def test_replaces_only_requested_part_and_refreshes_relative_pose(self):
        base = _trajectory(1.0)
        source = _trajectory(4.0)
        original_moving = copy.deepcopy(
            base["frames"]["000001"]["parts"]["moving"]
        )

        result, audit = compose_trajectory_parts(base, {"body": source})

        self.assertEqual(
            result["frames"]["000001"]["parts"]["moving"]["T_world_from_part"],
            original_moving["T_world_from_part"],
        )
        self.assertEqual(
            result["frames"]["000001"]["parts"]["body"]
            ["T_world_from_part"][0][3],
            4.0,
        )
        self.assertEqual(result["scales"]["body"], 6.0)
        self.assertEqual(audit["parts"]["body"]["replaced_frames"], 1)


if __name__ == "__main__":
    unittest.main()
