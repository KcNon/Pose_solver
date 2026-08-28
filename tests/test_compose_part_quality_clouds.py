from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from common.io_utils import load_json, write_json
from tools.stages.depth.compose_part_quality_clouds import compose_part_roots


class ComposePartQualityCloudsTests(unittest.TestCase):
    def test_composes_overlapping_frames_by_part(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = {}
            for part in ("moving", "body"):
                source = root / part
                sources[part] = source
                write_json(source / "quality_cloud_summary.json", {
                    "views": ["left", "right"],
                    "parameters": {"part": part},
                    "frames": {
                        "000001": {part: {"status": "ok"}},
                        "000002": {part: {"status": "ok"}},
                    },
                })
                for frame in (1, 2):
                    path = source / f"{frame:06d}" / f"{part}.ply"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(part, encoding="utf-8")

            output = root / "composed"
            report = compose_part_roots(
                sources, output, frame_start=1, frame_end=2
            )

            self.assertEqual(report["copied_frame_clouds"], {
                "moving": 2,
                "body": 2,
            })
            self.assertEqual(
                set(load_json(output / "quality_cloud_summary.json")
                    ["frames"]["000001"]),
                {"moving", "body"},
            )
            self.assertTrue((output / "000002" / "body.ply").exists())


if __name__ == "__main__":
    unittest.main()
