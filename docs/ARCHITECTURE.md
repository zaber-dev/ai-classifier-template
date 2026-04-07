# Architecture

## Goal

Provide a clean, local-first template for classification projects with clear extension points and maintainable boundaries.

## Module Boundaries

- `ai_classifier.core`
  - Contracts, configuration models, and exceptions.
  - Should remain dependency-light and stable.

- `ai_classifier.data`
  - Data loading and preprocessing helpers.
  - Owns conversion from raw CSV to numeric feature vectors.

- `ai_classifier.classifiers`
  - Model implementations behind common classifier interface.
  - Each classifier controls its own training and prediction logic.

- `ai_classifier.training`
  - Pipeline orchestration and metric computation.
  - Should not contain model-specific assumptions.

- `ai_classifier.utils`
  - Cross-cutting concerns: serialization and logging.

- `ai_classifier.cli`
  - Runtime interface for local workflows.
  - No core business logic should live here.

## Data Flow

1. Config is loaded and validated.
2. Dataset is loaded from CSV.
3. Data is split into train/test partitions.
4. Classifier is instantiated from model kind.
5. Model is fit on train data.
6. Predictions are generated for test data.
7. Metrics and artifacts are saved.

## Extension Strategy

To add new capabilities without destabilizing core behavior:

1. Add new classifier modules implementing `BaseClassifier`.
2. Register new model kind in pipeline factory.
3. Expand serialization if needed.
4. Add unit and integration tests.
5. Add docs and examples.

## Dependency Philosophy

- Keep baseline dependencies minimal.
- Place heavy frameworks into optional extras.
- Preserve local execution for all default workflows.
