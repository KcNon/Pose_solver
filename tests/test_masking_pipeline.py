from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from common.masking.compose import compose_frame, compose_track_tree
from common.masking.io import load_label_mask, save_binary_mask, track_path
from common.masking.multiview import project_mask_to_view
from common.masking.planning import (
    choose_seed_frames,
    detection_evidence,
    discovery_timestamps,
    infer_presence_start,
    repair_jobs_from_quality,
    resolve_mask_config,
)
from common.masking.quality import summarize_area_series, summarize_track_series
from common.masking.sam import normalized_xyxy_to_xywh
from common.masking.schema import load_mask_pipeline_config
from scripts.run_mask_pipeline import (
    _coalesce_sam_jobs,
    _job_bbox_fingerprint,
    _sam_jobs,
    _seed_frames,
)
from tools.stages.masking.track_part_masks import (
    _anchor_segments,
    _seed_for,
    _tracking_window,
    _trusted_seed_candidates,
    _validated_seed,
    _visibility_reference,
)
from tools.stages.masking.detect_mask_seeds import (
    _assign_candidates_from_mesh,
    _canonicalize_candidate_labels,
    _request_fingerprint,
    _unique_candidate_boxes,
)
from tools.stages.masking.validate_multiview_seeds import (
    _ambiguous_duplicate_parts,
    _box_iou,
)


class MaskSchemaTests(unittest.TestCase):
    def write_config(self, root: Path, data: dict) -> Path:
        path = root / "mask.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_discovery_timestamps_can_limit_the_coarse_search_window(self):
        timestamps = [f"{frame:06d}" for frame in range(0, 28)]
        self.assertEqual(
            discovery_timestamps(timestamps, stride=10, maximum_frame=24),
            ["000000", "000010", "000020", "000024"],
        )

    def test_visibility_reference_excludes_nearly_occluded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = root / "whole"
            for timestamp, pixels in (("000001", 100), ("000002", 80), ("000003", 5)):
                mask = np.zeros((10, 10), dtype=bool)
                mask.flat[:pixels] = True
                save_binary_mask(masks / timestamp / "cam.png", mask)
            config = SimpleNamespace(raw={"mask_quality": {
                "visibility_reference_masks": str(masks),
                "visibility_reference_minimum_pixels": 1,
                "visibility_reference_minimum_median_fraction": 0.5,
            }})
            eligible, report = _visibility_reference(
                config, "cam", ["000001", "000002", "000003"]
            )
            self.assertEqual(eligible, {"000001", "000002"})
            self.assertEqual(report["threshold_pixels"], 40)

    def test_arbitrary_parts_and_stable_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "work_root": str(root / "work"),
                "output_root": str(root / "output"),
                "views": ["left", "right"],
                "parts": {
                    "base": {
                        "id": 7,
                        "color": [1, 2, 3],
                        "start_frame": 12,
                        "prompts": ["object base"],
                    },
                    "handle": {
                        "id": 9,
                        "color": [4, 5, 6],
                        "start_frame": 20,
                        "prompts": ["handle"],
                    },
                },
                "occlusion_order": ["handle", "base"],
            }))
            self.assertEqual(config.part_names, ["base", "handle"])
            self.assertEqual(config.part_map["base"].id, 7)
            self.assertEqual(config.part_map["handle"].start_frame, 20)
            self.assertEqual(config.occlusion_order, ("handle", "base"))

    def test_duplicate_part_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": {
                    "one": {"id": 2},
                    "two": {"id": 2},
                },
                "occlusion_order": ["one", "two"],
            })
            with self.assertRaises(ValueError):
                load_mask_pipeline_config(path)

    def test_automatic_ids_do_not_collide_with_legacy_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": {
                    "custom": {},
                    "lid": {},
                    "body": {},
                },
                "occlusion_order": ["custom", "lid", "body"],
            }))
            ids = [part.id for part in config.parts]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(config.part_map["lid"].id, 1)
            self.assertEqual(config.part_map["body"].id, 2)
            self.assertEqual(config.part_map["custom"].id, 3)

    def test_per_view_seeds_and_segment_view_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["left", "right"],
                "parts": {
                    "body": {
                        "start_frame": 4,
                        "tracking": {
                            "mode": "fixed-image",
                            "seed_frames": {"default": 10, "right": 20},
                        },
                    },
                    "lid": {
                        "start_frame": 8,
                        "tracking": {
                            "mode": "video",
                            "seed_frame": 12,
                            "segments": [{
                                "views": ["right"],
                                "range": [8, 9],
                                "seed_frame": 9,
                            }],
                        },
                    },
                },
                "occlusion_order": ["lid", "body"],
            }))
            self.assertEqual(_seed_for(config, "body", "left", None), "000010")
            self.assertEqual(_seed_for(config, "body", "right", None), "000020")
            self.assertEqual(
                _seed_frames(config), ["000009", "000010", "000012", "000020"]
            )
            jobs = _sam_jobs(
                config,
                ["000000", "000001", "000002", "000003", "000004",
                 "000005", "000006", "000007", "000008", "000009",
                 "000010", "000011", "000012"],
                ["lid"],
                ["left"],
            )
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["views"], ["left"])

    def test_legacy_list_tracking_defaults_are_name_agnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": ["base", "piece"],
                "part_start_frames": {"base": 2, "piece": 3},
                "fixed_image_parts": ["base"],
                "occlusion_order": ["piece", "base"],
            }))
            jobs = _sam_jobs(
                config,
                [f"{frame:06d}" for frame in range(6)],
                ["base", "piece"],
            )
            self.assertEqual(
                {job["part"]: job["mode"] for job in jobs},
                {"base": "fixed-image", "piece": "video"},
            )

    def test_video_tracking_window_includes_seed_and_requested_range(self):
        frames = [f"{frame:06d}" for frame in range(20)]
        window, offset = _tracking_window(
            frames,
            ["000005", "000006", "000007"],
            "000010",
        )
        self.assertEqual(offset, 5)
        self.assertEqual(window, frames[5:11])

    def test_qwen_box_conversion_for_instance_seed(self):
        np.testing.assert_allclose(
            normalized_xyxy_to_xywh([100, 200, 600, 800]),
            [0.1, 0.2, 0.5, 0.6],
        )

    def test_explicit_config_can_disable_mesh_seed_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "meshes").mkdir()
            (root / "meshes" / "part.glb").write_bytes(b"mesh")
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {"start_frame": 10}},
                "occlusion_order": ["part"],
                "require_mesh_assignment": False,
            }))
            bbox = {"frames": {"000010": {"cam": {
                "parts": [{"label": "part", "bbox_2d": [1, 2, 3, 4]}],
                "mesh_assignment": {"status": "disabled"},
            }}}}
            self.assertEqual(
                _validated_seed(
                    config, bbox, ["000010"], "part", "cam", "000010"
                ),
                ("000010", "configured"),
            )

    def test_qwen_seed_window_expands_from_authoritative_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {
                    "start_frame": 10,
                    "tracking": {"seed_frame": 10},
                }},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 12, "stride": 5},
            }))
            self.assertEqual(
                _seed_frames(config),
                ["000010", "000015", "000020"],
            )

    def test_selected_part_includes_later_part_event_windows_for_reanchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {
                    "early": {
                        "start_frame": 10,
                        "tracking": {"seed_frame": 10},
                    },
                    "later": {
                        "start_frame": 30,
                        "tracking": {"seed_frame": 30},
                    },
                },
                "occlusion_order": ["later", "early"],
                "qwen_seed_window": {"length": 5, "stride": 5},
                "qwen_reanchor_on_part_events": True,
            }))
            self.assertEqual(
                _seed_frames(config, ["early"]),
                ["000010", "000015", "000030", "000035"],
            )

    def test_periodic_qwen_reanchors_extend_to_sequence_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "frames" / "cam"
            camera.mkdir(parents=True)
            for frame in range(251):
                (camera / f"{frame:06d}.jpg").touch()
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {
                    "start_frame": 10,
                    "tracking": {"seed_frame": 10},
                }},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 5, "stride": 5},
                "qwen_reanchor_on_part_events": True,
                "qwen_periodic_stride": 100,
            }))
            self.assertEqual(
                _seed_frames(config, ["part"]),
                ["000010", "000015", "000110", "000210"],
            )

    def test_seed_window_ignores_stale_bbox_outside_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {
                    "start_frame": 10,
                    "tracking": {"seed_frame": 10},
                }},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 10, "stride": 5},
                "require_mesh_assignment": False,
            }))
            frames = [f"{frame:06d}" for frame in range(31)]
            bbox = {"frames": {
                timestamp: {"cam": {"parts": [{"label": "part"}]}}
                for timestamp in ("000010", "000015", "000025")
            }}
            self.assertEqual(
                _trusted_seed_candidates(
                    config, bbox, frames, frames[10:], "part", "cam", "000010"
                ),
                ["000010", "000015"],
            )

    def test_part_event_reanchor_accepts_later_visual_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {
                    "start_frame": 10,
                    "tracking": {"seed_frame": 10},
                }},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 10, "stride": 5},
                "qwen_reanchor_on_part_events": True,
                "require_mesh_assignment": False,
            }))
            frames = [f"{frame:06d}" for frame in range(31)]
            bbox = {"frames": {
                timestamp: {"cam": {"parts": [{"label": "part"}]}}
                for timestamp in ("000010", "000015", "000025")
            }}
            self.assertEqual(
                _trusted_seed_candidates(
                    config, bbox, frames, frames[10:], "part", "cam", "000010"
                ),
                ["000010", "000015", "000025"],
            )

    def test_adaptive_initial_anchor_prefers_multiview_complete_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            views = [f"cam{index}" for index in range(4)]
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": views,
                "parts": {"part": {
                    "start_frame": 10,
                    "tracking": {"seed_frame": 10},
                }},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 10, "stride": 5},
                "qwen_reanchor_on_part_events": True,
                "qwen_initial_anchor_selection": {
                    "enabled": True,
                    "minimum_views": 2,
                    "minimum_view_fraction": 0.5,
                    "reject_border_clipped": True,
                },
                "require_mesh_assignment": False,
            }))
            frames = [f"{frame:06d}" for frame in range(31)]
            bbox = {"frames": {
                "000010": {"cam0": {"parts": [
                    {"label": "part", "bbox_2d": [100, 100, 190, 190]}
                ]}},
                "000015": {view: {"parts": [
                    {"label": "part", "bbox_2d": [100, 100, 220, 220]}
                ]} for view in views},
                "000020": {view: {"parts": [
                    {"label": "part", "bbox_2d": [100, 100, 180, 180]}
                ]} for view in views},
                "000025": {"cam0": {"parts": [
                    {"label": "part", "bbox_2d": [100, 100, 200, 200]}
                ]}},
            }}
            self.assertEqual(
                _trusted_seed_candidates(
                    config, bbox, frames, frames[10:], "part", "cam0",
                    "000010", views,
                ),
                ["000015", "000025"],
            )

    def test_adaptive_initial_anchor_rejects_border_clipped_bbox(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            views = ["left", "right"]
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": views,
                "parts": {"part": {"start_frame": 10}},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 5, "stride": 5},
                "qwen_reanchor_on_part_events": True,
                "qwen_initial_anchor_selection": {
                    "enabled": True,
                    "minimum_views": 2,
                    "reject_border_clipped": True,
                },
                "require_mesh_assignment": False,
            }))
            frames = [f"{frame:06d}" for frame in range(20)]
            bbox = {"frames": {
                "000010": {view: {"parts": [
                    {"label": "part", "bbox_2d": [0, 100, 200, 250]}
                ]} for view in views},
                "000015": {view: {"parts": [
                    {"label": "part", "bbox_2d": [100, 100, 300, 250]}
                ]} for view in views},
            }}
            self.assertEqual(
                _trusted_seed_candidates(
                    config, bbox, frames, frames[10:], "part", "left",
                    "000010", views,
                ),
                ["000015"],
            )

    def test_adaptive_initial_anchor_respects_delay_and_sharpness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views = ["left", "right"]
            for view in views:
                camera = root / "frames" / view
                camera.mkdir(parents=True)
                checker = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(
                    np.uint8
                )
                Image.fromarray(checker).save(camera / "000010.jpg")
                Image.fromarray(np.full((100, 100), 128, np.uint8)).save(
                    camera / "000015.jpg"
                )
                Image.fromarray(checker).save(camera / "000020.jpg")
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": views,
                "parts": {"part": {"start_frame": 10}},
                "occlusion_order": ["part"],
                "qwen_seed_window": {"length": 10, "stride": 5},
                "qwen_reanchor_on_part_events": True,
                "qwen_initial_anchor_selection": {
                    "enabled": True,
                    "minimum_delay": 5,
                    "minimum_views": 2,
                    "reject_border_clipped": True,
                },
                "require_mesh_assignment": False,
            }))
            frames = [f"{frame:06d}" for frame in range(10, 21)]
            record = lambda: {"parts": [
                {"label": "part", "bbox_2d": [100, 100, 900, 900]}
            ]}
            bbox = {"frames": {
                timestamp: {view: record() for view in views}
                for timestamp in ("000010", "000015", "000020")
            }}
            self.assertEqual(
                _trusted_seed_candidates(
                    config, bbox, frames, frames, "part", "left",
                    "000010", views,
                ),
                ["000020"],
            )

    def test_qwen_fingerprint_changes_with_mesh_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.jpg"
            reference = root / "preview.jpg"
            image.write_bytes(b"frame")
            reference.write_bytes(b"first")
            kwargs = {
                "image_path": image,
                "prompt": "find part",
                "model_path": "model",
                "max_new_tokens": 10,
                "max_pixels": 20,
                "reference_images": {"part": reference},
            }
            first = _request_fingerprint(**kwargs)
            reference.write_bytes(b"second")
            self.assertNotEqual(first, _request_fingerprint(**kwargs))

    def test_duplicate_qwen_boxes_become_one_mesh_candidate(self):
        boxes = [
            {"bbox_2d": [100, 100, 300, 300], "label": "one"},
            {"bbox_2d": [102, 102, 298, 298], "label": "two"},
            {"bbox_2d": [500, 500, 700, 700], "label": "two"},
        ]
        self.assertEqual(
            _unique_candidate_boxes(boxes),
            [[100.0, 100.0, 300.0, 300.0], [500.0, 500.0, 700.0, 700.0]],
        )

    def test_parser_normalized_label_maps_back_to_mesh_part_name(self):
        self.assertEqual(
            _canonicalize_candidate_labels(
                [{"bbox_2d": [1, 2, 3, 4], "label": "whole_close"}],
                {"whole-close"},
                keep_unmatched=False,
            ),
            [{"bbox_2d": [1, 2, 3, 4], "label": "whole-close"}],
        )

    def test_mesh_assignment_preserves_supported_unique_qwen_labels(self):
        boxes = [
            {"bbox_2d": [10, 10, 40, 40], "label": "main"},
            {"bbox_2d": [50, 50, 80, 80], "label": "collector"},
        ]
        # Rows: collector reference, main reference, then two candidates.
        # A purely global assignment would swap these proposals by a small
        # margin, despite both Qwen names having mesh support above threshold.
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2073, 0.2081],
            [0.4102, 0.4576],
        ])
        parts = [SimpleNamespace(name="collector"), SimpleNamespace(name="main")]
        with (
            patch(
                "tools.stages.masking.detect_mask_seeds._preview_views",
                return_value=[object()],
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._target_crop",
                return_value=object(),
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._dino_embeddings",
                return_value=vectors,
            ),
        ):
            assigned, report = _assign_candidates_from_mesh(
                object(), boxes, parts,
                {"collector": Path("collector"), "main": Path("main")},
                object(), object(),
            )
        self.assertEqual(
            {row["label"]: row["bbox_2d"] for row in assigned},
            {
                "main": [10.0, 10.0, 40.0, 40.0],
                "collector": [50.0, 50.0, 80.0, 80.0],
            },
        )
        self.assertEqual(
            set(report["qwen_locked_parts"]), {"collector", "main"}
        )

    def test_mesh_assignment_can_correct_unsupported_qwen_label(self):
        boxes = [{"bbox_2d": [10, 10, 40, 40], "label": "collector"}]
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.18, 0.21],
        ])
        parts = [SimpleNamespace(name="collector"), SimpleNamespace(name="main")]
        with (
            patch(
                "tools.stages.masking.detect_mask_seeds._preview_views",
                return_value=[object()],
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._target_crop",
                return_value=object(),
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._dino_embeddings",
                return_value=vectors,
            ),
        ):
            assigned, report = _assign_candidates_from_mesh(
                object(), boxes, parts,
                {"collector": Path("collector"), "main": Path("main")},
                object(), object(),
            )
        self.assertEqual(assigned[0]["label"], "main")
        self.assertEqual(report["qwen_locked_parts"], [])

    def test_single_qwen_label_must_also_be_mesh_best(self):
        boxes = [{"bbox_2d": [10, 10, 40, 40], "label": "collector"}]
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.21, 0.28],
        ])
        parts = [SimpleNamespace(name="collector"), SimpleNamespace(name="main")]
        with (
            patch(
                "tools.stages.masking.detect_mask_seeds._preview_views",
                return_value=[object()],
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._target_crop",
                return_value=object(),
            ),
            patch(
                "tools.stages.masking.detect_mask_seeds._dino_embeddings",
                return_value=vectors,
            ),
        ):
            assigned, report = _assign_candidates_from_mesh(
                object(), boxes, parts,
                {"collector": Path("collector"), "main": Path("main")},
                object(), object(),
            )
        self.assertEqual(assigned[0]["label"], "main")
        self.assertEqual(report["qwen_locked_parts"], [])

    def test_video_seed_falls_back_to_nearest_mesh_validated_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "meshes").mkdir()
            (root / "meshes" / "part.glb").write_bytes(b"mesh")
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {"start_frame": 0}},
                "occlusion_order": ["part"],
            }))
            bbox = {"frames": {
                "000010": {"cam": {"parts": []}},
                "000014": {"cam": {
                    "parts": [{"label": "part", "bbox_2d": [1, 2, 3, 4]}],
                    "mesh_assignment": {"status": "ok"},
                }},
            }}
            self.assertEqual(
                _validated_seed(
                    config,
                    bbox,
                    ["000010", "000014"],
                    "part",
                    "cam",
                    "000010",
                ),
                ("000014", "nearest_mesh_validated"),
            )

    def test_mesh_validated_seeds_create_nearest_anchor_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            (root / "meshes").mkdir()
            (root / "meshes" / "part.glb").write_bytes(b"mesh")
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "views": ["cam"],
                "parts": {"part": {"start_frame": 0}},
                "occlusion_order": ["part"],
            }))
            frames = [f"{frame:06d}" for frame in range(10)]
            bbox = {"frames": {
                "000002": {"cam": {
                    "parts": [{"label": "part"}],
                    "mesh_assignment": {"status": "ok"},
                }},
                "000005": {"cam": {
                    "parts": [{"label": "part"}],
                    "mesh_assignment": {"status": "unavailable"},
                }},
                "000007": {"cam": {
                    "parts": [{"label": "part"}],
                    "mesh_assignment": {"status": "ok"},
                }},
            }}
            anchors = _trusted_seed_candidates(
                config, bbox, frames, frames, "part", "cam", "000005"
            )
            self.assertEqual(anchors, ["000002", "000007"])
            segments = _anchor_segments(frames, frames, anchors)
            self.assertEqual(
                [segment["requested"] for segment in segments],
                [frames[:5], frames[5:]],
            )
            self.assertEqual(segments[0]["window_ids"], frames[:5])
            self.assertEqual(segments[1]["window_ids"], frames[5:])

    def test_equal_sam_repair_jobs_share_one_model_process(self):
        common = {
            "part": "piece",
            "mode": "video",
            "range": [10, 14],
            "seed_frame": "000012",
            "hold_previous": False,
            "repair": True,
        }
        grouped = _coalesce_sam_jobs([
            {**common, "views": ["left"]},
            {**common, "views": ["right"]},
        ])
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["views"], ["left", "right"])

    def test_sam_fingerprint_ignores_unrelated_bbox_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_mask_pipeline_config(self.write_config(root, {
                "frames_dir": str(root),
                "views": ["cam"],
                "parts": {
                    "back": {
                        "start_frame": 5,
                        "tracking": {"seed_frame": 5},
                    },
                    "front": {"start_frame": 5},
                },
                "occlusion_order": ["front", "back"],
            }))
            job = {
                "part": "back",
                "mode": "video",
                "views": ["cam"],
                "range": [5, 10],
                "seed_frame": "000005",
            }
            bbox = {"frames": {"000005": {"cam": {"parts": [
                {"label": "back", "bbox_2d": [1, 2, 3, 4]},
            ]}}}}
            first = _job_bbox_fingerprint(
                config, job, ["000005"], bbox
            )
            bbox["frames"]["000009"] = {
                "cam": {"parts": [
                    {"label": "front", "bbox_2d": [4, 3, 2, 1]},
                ]}
            }
            bbox["frames"]["000005"]["cam"]["parts"].append(
                {"label": "front", "bbox_2d": [5, 6, 7, 8]}
            )
            second = _job_bbox_fingerprint(
                config, job, ["000005"], bbox
            )
            self.assertEqual(first, second)
            bbox["frames"]["000005"]["cam"]["parts"][0]["bbox_2d"][0] = 9
            self.assertNotEqual(
                first,
                _job_bbox_fingerprint(config, job, ["000005"], bbox),
            )
            later = _job_bbox_fingerprint(config, job, ["000005"], bbox)
            bbox["frames"]["000009"]["cam"]["parts"].append(
                {"label": "back", "bbox_2d": [20, 30, 40, 50]}
            )
            self.assertNotEqual(
                later,
                _job_bbox_fingerprint(config, job, ["000005"], bbox),
            )

    def test_auto_start_is_unresolved_until_planning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_config(root, {
                "frames_dir": str(root / "frames"),
                "work_root": str(root / "work"),
                "views": ["left", "right"],
                "parts": {
                    "part": {
                        "start_frame": "auto",
                        "tracking": {
                            "mode": "video",
                            "seed_frame": "auto",
                        },
                    },
                },
                "occlusion_order": ["part"],
            })
            config = load_mask_pipeline_config(path)
            self.assertTrue(config.part_map["part"].start_frame_auto)
            bbox = {
                "frames": {
                    "000000": {
                        "left": {"parts": []},
                        "right": {"parts": []},
                    },
                    "000010": {
                        "left": {"parts": [{
                            "label": "part",
                            "bbox_2d": [100, 100, 500, 500],
                        }]},
                        "right": {"parts": [{
                            "label": "part",
                            "bbox_2d": [200, 100, 600, 600],
                        }]},
                    },
                    "000020": {
                        "left": {"parts": [{
                            "label": "part",
                            "bbox_2d": [100, 100, 500, 500],
                        }]},
                        "right": {"parts": [{
                            "label": "part",
                            "bbox_2d": [200, 100, 600, 600],
                        }]},
                    },
                },
            }
            resolved_path = root / "resolved.json"
            raw, report = resolve_mask_config(
                config,
                bbox,
                ["000000", "000010", "000020"],
                output_path=resolved_path,
            )
            self.assertEqual(raw["parts"]["part"]["start_frame"], 10)
            self.assertEqual(
                set(raw["parts"]["part"]["tracking"]["seed_frames"]),
                {"left", "right"},
            )
            self.assertEqual(
                report["parts"]["part"]["start_source"],
                "qwen_multiview_discovery",
            )


class MaskCompositionTests(unittest.TestCase):
    def config(self, root: Path):
        path = root / "mask.json"
        path.write_text(json.dumps({
            "frames_dir": str(root / "frames"),
            "views": ["cam"],
            "parts": {
                "back": {"id": 4, "start_frame": 5, "color": [0, 255, 0]},
                "front": {"id": 8, "start_frame": 10, "color": [255, 0, 0]},
            },
            "occlusion_order": ["front", "back"],
        }), encoding="utf-8")
        return load_mask_pipeline_config(path)

    def test_start_frames_and_front_to_back_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            back = np.zeros((5, 5), bool)
            front = np.zeros((5, 5), bool)
            back[1:4, 1:4] = True
            front[2:5, 2:5] = True
            label, resolved = compose_frame(
                {"back": back, "front": front}, config, 7, (5, 5)
            )
            self.assertFalse(resolved["front"].any())
            self.assertEqual(int((label == 4).sum()), 9)
            label, resolved = compose_frame(
                {"back": back, "front": front}, config, 10, (5, 5)
            )
            self.assertEqual(int((label == 8).sum()), 9)
            self.assertEqual(int((label == 4).sum()), 5)
            self.assertFalse((resolved["front"] & resolved["back"]).any())

    def test_quality_report_suggests_middle_of_empty_run(self):
        report = summarize_area_series(
            ["000004", "000005", "000006", "000007", "000008"],
            [0, 100, 0, 0, 100],
            start_frame=5,
        )
        self.assertEqual(report["empty_runs"], [[6, 7]])
        self.assertEqual(report["suggested_reanchor_frames"], [7])

    def test_track_quality_requires_jump_for_low_iou(self):
        report = summarize_track_series(
            ["000005", "000006", "000007"],
            [100, 100, 100],
            [[10, 10], [11, 10], [90, 90]],
            [None, 0.01, 0.01],
            start_frame=5,
            image_diagonal=100.0,
            max_centroid_step_ratio=0.2,
        )
        self.assertEqual(report["low_temporal_iou_frames"], [7])

    def test_track_quality_preserves_full_empty_run_for_repair(self):
        report = summarize_track_series(
            ["000005", "000006", "000007", "000008", "000009"],
            [100, 0, 0, 0, 100],
            [[10, 10], None, None, None, [11, 10]],
            [None, 0.0, 1.0, 1.0, 0.0],
            start_frame=5,
            image_diagonal=100.0,
        )
        self.assertEqual(report["anomaly_runs"], [[6, 8]])
        self.assertEqual(report["suggested_reanchor_frames"], [7])

    def test_track_tree_enforces_active_track_completeness(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            frame_root = root / "frames" / "cam"
            frame_root.mkdir(parents=True)
            for timestamp in ("000004", "000005", "000006"):
                Image.new("RGB", (8, 6)).save(frame_root / f"{timestamp}.jpg")
            back = np.zeros((6, 8), bool)
            back[1:4, 2:6] = True
            save_binary_mask(
                track_path(config.tracks_root, "back", "000005", "cam"),
                back,
            )
            # Missing tracks are allowed before start_frame.
            compose_track_tree(config, ["000004", "000005"])
            label = load_label_mask(config.masks_root / "000005" / "cam.png")
            self.assertEqual(int((label == 4).sum()), int(back.sum()))
            # Once a part exists, a missing file is an incomplete run, not
            # evidence that the object is invisible.
            with self.assertRaises(FileNotFoundError):
                compose_track_tree(config, ["000006"])

    def test_track_tree_can_fill_closed_texture_holes(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            config.raw["mask_postprocess"] = {
                "enabled": True,
                "close_kernel": 1,
                "fill_holes": True,
            }
            frame_root = root / "frames" / "cam"
            frame_root.mkdir(parents=True)
            Image.new("RGB", (8, 6)).save(frame_root / "000005.jpg")
            back = np.zeros((6, 8), bool)
            back[1:5, 1:7] = True
            back[2, 3] = False
            save_binary_mask(
                track_path(config.tracks_root, "back", "000005", "cam"),
                back,
            )
            compose_track_tree(config, ["000005"])
            label = load_label_mask(config.masks_root / "000005" / "cam.png")
            self.assertEqual(int(label[2, 3]), 4)

    def test_repair_jobs_merge_padded_anomaly_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            config.raw["automation"] = {
                "repair": {
                    "padding_frames": 1,
                    "maximum_jobs_per_part": 3,
                }
            }
            config.raw["parts"]["back"]["tracking"] = {
                "mode": "fixed-image"
            }
            quality = {
                "cam": {
                    "back": {"anomaly_runs": [[6, 6], [8, 8]]},
                    "front": {"anomaly_runs": []},
                }
            }
            jobs = repair_jobs_from_quality(
                quality,
                config,
                [f"{frame:06d}" for frame in range(5, 11)],
            )
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["range"], [5, 9])
            self.assertEqual(jobs[0]["mode"], "fixed-image")


class MultiViewMaskTests(unittest.TestCase):
    def test_cross_part_duplicate_boxes_are_ambiguous(self):
        bbox_data = {
            "frames": {
                "000010": {
                    "cam": {
                        "parts": [
                            {"label": "body", "bbox_2d": [10, 20, 100, 200]},
                            {"label": "nozzle", "bbox_2d": [10, 20, 100, 200]},
                        ]
                    }
                }
            }
        }
        self.assertEqual(
            _ambiguous_duplicate_parts(
                bbox_data,
                "000010",
                "cam",
                ["body", "nozzle"],
                overrides=None,
                iou_threshold=0.95,
            ),
            {"body", "nozzle"},
        )

    def test_distinct_touching_part_boxes_remain_valid(self):
        self.assertLess(
            _box_iou([10, 20, 100, 200], [80, 20, 170, 200]),
            0.95,
        )

    def test_identity_cameras_preserve_mask(self):
        depth = np.full((20, 20), 2.0, np.float32)
        mask = np.zeros((20, 20), bool)
        mask[6:14, 7:13] = True
        intrinsic = np.array([
            [20.0, 0.0, 10.0],
            [0.0, 20.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        projected = project_mask_to_view(
            mask,
            depth,
            intrinsic,
            extrinsic,
            depth,
            intrinsic,
            extrinsic,
            depth_tolerance=1e-4,
            dilate=0,
            close_kernel=1,
        )
        np.testing.assert_array_equal(projected, mask)

    def test_target_occluder_rejects_projected_surface(self):
        source_depth = np.full((20, 20), 2.0, np.float32)
        target_depth = np.full((20, 20), 1.0, np.float32)
        mask = np.zeros((20, 20), bool)
        mask[6:14, 7:13] = True
        intrinsic = np.array([
            [20.0, 0.0, 10.0],
            [0.0, 20.0, 10.0],
            [0.0, 0.0, 1.0],
        ])
        extrinsic = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
        projected = project_mask_to_view(
            mask,
            source_depth,
            intrinsic,
            extrinsic,
            target_depth,
            intrinsic,
            extrinsic,
            depth_tolerance=0.05,
            dilate=0,
            close_kernel=1,
        )
        self.assertFalse(projected.any())


if __name__ == "__main__":
    unittest.main()
