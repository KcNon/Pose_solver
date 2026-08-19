import unittest

from tools.stages.preprocess.extract_synchronized_video_frames import (
    parse_frame_rate,
)


class PreprocessTimelineTest(unittest.TestCase):
    def test_rational_native_frame_rate(self):
        self.assertAlmostEqual(parse_frame_rate("30000/1001"), 29.97002997)

    def test_invalid_native_frame_rate(self):
        with self.assertRaises(ValueError):
            parse_frame_rate("0/1")


if __name__ == "__main__":
    unittest.main()
