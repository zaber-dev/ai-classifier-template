from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


@dataclass(slots=True)
class DatasetConfig:
    path: str
    label_column: str
    test_size: float = 0.2
    shuffle: bool = True
    random_seed: int = 42

    def validate(self) -> None:
        if not self.path:
            raise ConfigurationError("dataset.path is required")
        if not Path(self.path).exists():
            raise ConfigurationError(f"Dataset file not found: {self.path}")
        if not self.label_column:
            raise ConfigurationError("dataset.label_column is required")
        if not 0.0 < self.test_size < 1.0:
            raise ConfigurationError("dataset.test_size must be between 0 and 1")


@dataclass(slots=True)
class ModelConfig:
    kind: str = "template"
    params: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.kind not in {"template", "sklearn"}:
            raise ConfigurationError("model.kind must be either 'template' or 'sklearn'")


@dataclass(slots=True)
class TrainingConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )

    def validate(self) -> None:
        allowed = {"accuracy", "precision", "recall", "f1"}
        unknown = [metric for metric in self.metrics if metric not in allowed]
        if unknown:
            raise ConfigurationError(f"Unsupported metrics requested: {unknown}")


@dataclass(slots=True)
class OutputConfig:
    model_path: str = "artifacts/model.json"
    report_path: str = "artifacts/report.json"

    def validate(self) -> None:
        if not self.model_path:
            raise ConfigurationError("output.model_path is required")
        if not self.report_path:
            raise ConfigurationError("output.report_path is required")


@dataclass(slots=True)
class PipelineConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        dataset = DatasetConfig(**raw.get("dataset", {}))
        model = ModelConfig(**raw.get("model", {}))
        training = TrainingConfig(**raw.get("training", {}))
        output = OutputConfig(**raw.get("output", {}))
        config = cls(dataset=dataset, model=model, training=training, output=output)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise ConfigurationError(f"Config file not found: {path}")
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("Config file must resolve to a mapping")
        return cls.from_dict(raw)

    def validate(self) -> None:
        self.dataset.validate()
        self.model.validate()
        self.training.validate()
        self.output.validate()
