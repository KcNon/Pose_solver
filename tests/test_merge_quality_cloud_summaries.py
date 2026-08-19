from pathlib import Path
import tempfile
import unittest

from common.io_utils import write_json
from tools.diagnostics.merge_quality_cloud_summaries import merge_summaries


class MergeQualityCloudSummariesTest(unittest.TestCase):
    def _summary(self, frames):
        return {
            "schema_version": 2,
            "method": "quality",
            "depth_gauge": None,
            "output_root": "/tmp/clouds",
            "views": ["a", "b"],
            "parameters": {"threshold": 1},
            "frames": {str(frame): {"part": {"status": "ok"}} for frame in frames},
        }

    def test_merges_and_sorts_disjoint_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, self._summary([2, 3]))
            write_json(second, self._summary([0, 1]))
            merged = merge_summaries([first, second])
        self.assertEqual(list(merged["frames"]), ["0", "1", "2", "3"])

    def test_rejects_overlapping_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, self._summary([0, 1]))
            write_json(second, self._summary([1, 2]))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                merge_summaries([first, second])


if __name__ == "__main__":
    unittest.main()
