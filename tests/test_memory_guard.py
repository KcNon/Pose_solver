from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common.resource_safety import MEMORY_GUARD_ACTIVE, require_memory_guard
from tools.diagnostics.run_with_memory_guard import (
    GIB,
    memory_limit_reasons,
    parse_gpu_rows,
    process_group_rss_bytes,
    read_meminfo,
)


class MemoryGuardTest(unittest.TestCase):
    def test_heavy_cli_requires_guard_marker(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "cannot run unguarded"):
                require_memory_guard("test-tool")
        with patch.dict("os.environ", {MEMORY_GUARD_ACTIVE: "1"}, clear=True):
            require_memory_guard("test-tool")

    def test_meminfo_is_converted_from_kib_to_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            path.write_text(
                "MemTotal: 1024 kB\n"
                "MemAvailable: 512 kB\n"
                "SwapTotal: 128 kB\n"
                "SwapFree: 32 kB\n",
                encoding="utf-8",
            )
            result = read_meminfo(path)
        self.assertEqual(result["MemAvailable"], 512 * 1024)
        self.assertEqual(result["SwapFree"], 32 * 1024)

    def test_gpu_rows_use_physical_indices(self) -> None:
        result = parse_gpu_rows("6, 1024, 4096, 80\n7, 2048, 3072, 90\n")
        self.assertEqual(result[6]["memory_free_mib"], 4096)
        self.assertEqual(result[7]["utilization_percent"], 90)

    def test_memory_limit_reasons_fail_closed(self) -> None:
        meminfo = {
            "MemTotal": 755 * GIB,
            "MemAvailable": 90 * GIB,
            "SwapTotal": 8 * GIB,
            "SwapFree": 0,
        }
        reasons = memory_limit_reasons(
            meminfo,
            40 * GIB,
            minimum_available_gib=128.0,
            maximum_process_rss_gib=32.0,
        )
        self.assertEqual(len(reasons), 2)
        self.assertIn("available RAM", reasons[0])
        self.assertIn("process-group RSS", reasons[1])

    def test_process_exit_race_does_not_crash_monitor(self) -> None:
        with patch.object(
            Path,
            "iterdir",
            return_value=iter([Path("/proc/123")]),
        ), patch.object(
            Path,
            "read_text",
            side_effect=ProcessLookupError(3, "process disappeared"),
        ):
            rss, pids = process_group_rss_bytes(123)
        self.assertEqual(rss, 0)
        self.assertEqual(pids, [])


if __name__ == "__main__":
    unittest.main()
