"""Public package for the reusable multi-view pose pipeline."""

from pose_solver.config import PipelineConfig, load_pipeline_config
from pose_solver.pipeline import PipelineRunner

__all__ = ["PipelineConfig", "PipelineRunner", "load_pipeline_config"]
__version__ = "0.2.0"
