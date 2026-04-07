# API Reference

## Core Contracts

### `BaseClassifier`

Required methods:

- `fit(features: list[list[float]], labels: list[str]) -> None`
- `predict(features: list[list[float]]) -> list[str]`
- `to_artifact() -> dict[str, Any]`
- `from_artifact(artifact: dict[str, Any]) -> BaseClassifier`

### `BaseDataLoader`

- `load() -> tuple[list[list[float]], list[str]]`

### `BaseEvaluator`

- `compute(actual: list[str], predicted: list[str]) -> dict[str, float]`

## Config Models

### `PipelineConfig`

Top-level config composed of:

- `dataset: DatasetConfig`
- `model: ModelConfig`
- `training: TrainingConfig`
- `output: OutputConfig`

Factory helpers:

- `PipelineConfig.from_dict(raw)`
- `PipelineConfig.from_yaml(path)`

## Training

### `TrainingPipeline`

- `run() -> dict[str, float]`

Behavior:

1. Load and split dataset.
2. Build classifier from config.
3. Fit and evaluate.
4. Save model and report artifacts.

## Serialization

### `save_model(model, path)`

Writes model artifact as JSON or joblib depending on model type and extension.

### `load_model(path)`

Loads artifact and returns a classifier instance ready for prediction.

## CLI

Entry command: `ai-classifier`

Subcommands:

- `train --config <yaml>`
- `predict --model <path> --input <csv> --output <csv>`
- `evaluate --model <path> --input <csv> --label-column <col> --output <json>`
