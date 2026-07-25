"""Reusable mask-pipeline primitives.

The package deliberately contains no command-line orchestration.  Qwen and
SAM run in different Python environments, while configuration, mask I/O,
composition, quality checks, and multi-view geometry remain shared.
"""

from .schema import MaskPipelineConfig, PartSpec, load_mask_pipeline_config

__all__ = ["MaskPipelineConfig", "PartSpec", "load_mask_pipeline_config"]
