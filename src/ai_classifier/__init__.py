"""AI Classifier template package.

This package provides a local-first, customizable framework for building and
extending classifier projects with clean abstractions and reproducible pipelines.
"""

from .core.config import PipelineConfig
from .training.pipeline import TrainingPipeline

__all__ = ["PipelineConfig", "TrainingPipeline"]
