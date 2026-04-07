# AI Classifier Template

Production-minded, local-first Python template for building customizable classification projects.

This repository is designed for developers who want a clean baseline that feels like an internal engineering starter kit: stable abstractions, reproducible training pipeline, clear documentation, quality gates, and straightforward extension points.

## Why this template exists

Most classifier repos either overfit to one model stack or under-document architecture decisions. This template balances both sides:

- Local-first execution with no hosted dependency requirements.
- Strong extension points for project-specific classifiers.
- Professional documentation structure suited for public OSS and team handoffs.
- Reproducible artifacts and CLI workflows for daily development.
- Optional deep-learning path without forcing heavyweight dependencies in core installs.

## Core Features

- Extensible classifier interface via base contracts.
- Built-in template classifier (`TemplateCentroidClassifier`) for fast customization.
- Optional sklearn adapter (`model.kind: sklearn`) for common production baselines.
- YAML-driven pipeline config for train/evaluate reproducibility.
- CLI commands for local train/predict/evaluate.
- Starter dataset and pre-trained artifact for immediate demos.
- CI-ready test/lint/type-check setup.

## Repository Layout

```
.
|-- src/ai_classifier/      # Package source
|-- tests/                  # Unit and integration tests
|-- examples/               # Configs, sample data, starter scripts
|-- docs/                   # Architecture, API, troubleshooting, examples docs
|-- .github/workflows/      # CI and release pipelines
|-- README.md               # Public overview and quick start
|-- LEARN.md                # Deep technical handbook
|-- LICENSE.md              # Human-readable license notes
|-- LICENSE                 # Canonical MIT license text
```

## Quick Start

### 1. Install locally

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[ml]
```

### 2. Train using starter config

```bash
ai-classifier train --config examples/configs/template_config.yaml
```

This generates:

- `artifacts/model.json`
- `artifacts/report.json`

### 3. Predict locally

```bash
ai-classifier predict --model artifacts/model.json --input examples/data/predict.csv --output artifacts/predictions.csv
```

### 4. Evaluate on labeled data

```bash
ai-classifier evaluate --model artifacts/model.json --input examples/data/train.csv --label-column label --output artifacts/eval.json
```

## Local and Customizable by Design

You can create a new classifier with minimal work:

1. Implement `BaseClassifier` methods (`fit`, `predict`, `to_artifact`, `from_artifact`).
2. Add your classifier module under `src/ai_classifier/classifiers/`.
3. Extend model factory logic in pipeline.
4. Add tests and a docs snippet in `LEARN.md`.

The template classifier exists as a readable baseline so teams can fork and evolve without hidden magic.

## Documentation Map

- `README.md`: overview + quick start.
- `LEARN.md`: architecture and extension handbook.
- `docs/ARCHITECTURE.md`: module boundaries and design rationale.
- `docs/API.md`: public contracts and expected behavior.
- `docs/TROUBLESHOOTING.md`: common failures and fixes.
- `docs/EXAMPLES.md`: practical usage patterns.

## Quality and Release

Recommended local checks:

```bash
pytest
ruff check .
mypy src
python -m build
```

GitHub workflows are included for CI validation and package publishing.

## Public Repository Notes

- Maintainer: `zaber-dev`
- Suggested repository name: `ai-classifier-template`
- Versioning model: Semantic Versioning (`MAJOR.MINOR.PATCH`)

## License

MIT License. See `LICENSE` and `LICENSE.md`.
