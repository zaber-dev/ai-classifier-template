# LEARN: AI Classifier Template Handbook

This guide explains how the template is organized, how to extend it for real projects, and how to keep it maintainable over time.

## 1. Architecture Overview

The repository follows a layered structure:

1. `core`: contracts, config validation, exceptions.
2. `data`: CSV loading and deterministic splitting.
3. `classifiers`: model implementations.
4. `training`: orchestration and metrics.
5. `utils`: serialization and logging.
6. `cli`: local execution interface.

This separation keeps domain logic independent from execution surfaces (CLI, scripts, CI).

## 2. Design Principles

### Local-first

Everything runs offline with local files and local artifacts. No API service is required.

### Explicit extension points

New model types are added through `BaseClassifier`, then wired into pipeline model selection.

### Reproducibility

Configs are YAML-based, include seed settings, and produce serialized model/report artifacts.

### Progressive dependency model

Core install is lightweight (`PyYAML`).
Optional extras add heavier stacks:

- `ml`: sklearn + joblib.
- `dev`: test and quality tooling.

## 3. Configuration Model

`PipelineConfig` validates four sections:

- `dataset`: path, label column, split settings.
- `model`: implementation kind and parameters.
- `training`: metric list.
- `output`: model/report output paths.

Example config:

```yaml
dataset:
  path: examples/data/train.csv
  label_column: label
  test_size: 0.25
  shuffle: true
  random_seed: 7

model:
  kind: template
  params: {}

training:
  metrics: [accuracy, precision, recall, f1]

output:
  model_path: artifacts/model.json
  report_path: artifacts/report.json
```

## 4. How to Add a New Classifier

### Step 1: Create classifier module

Add a new file in `src/ai_classifier/classifiers/`, for example `my_classifier.py`.

Implement:

- `fit(features, labels)`
- `predict(features)`
- `to_artifact()`
- `from_artifact()`

### Step 2: Register in pipeline factory

Update classifier selection logic in `training/pipeline.py` so `model.kind` can instantiate your class.

### Step 3: Add tests

Add unit tests for:

- fit/predict behavior
- edge cases
- serialization round-trip

### Step 4: Document it

Add usage snippet and configuration example in `docs/EXAMPLES.md` and this handbook.

## 5. Optional Deep Learning Path

This template intentionally keeps deep-learning code optional. If you want torch-based classifiers:

1. Add a torch classifier adapter implementing `BaseClassifier`.
2. Put torch in a dedicated optional dependency group.
3. Keep model artifact compatibility explicit by storing metadata with version fields.
4. Add integration tests that run only when optional dependencies are installed.

This keeps baseline setup fast while enabling advanced project variants.

## 6. Serialization Strategy

- Template classifier uses JSON artifact (`model.json`).
- sklearn classifier can be saved as JSON metadata or `.joblib` when joblib is available.

Recommendation for production forks:

- Keep artifact schema versioned (`artifact_version`).
- Store training data hash and config snapshot for auditability.

## 7. Testing Strategy

Minimum baseline:

1. Unit tests for each classifier and config validation.
2. Integration test for end-to-end training pipeline.
3. CLI smoke tests for train/predict/evaluate commands.

Recommended expansion:

1. Backward compatibility tests for artifacts.
2. Performance benchmarks on representative datasets.
3. Property-based tests for preprocessing and metrics.

## 8. Release and Maintenance

### Versioning

Use semantic versions:

- `MAJOR`: breaking changes.
- `MINOR`: backward-compatible feature additions.
- `PATCH`: fixes and non-breaking improvements.

### Release checklist

1. Update changelog.
2. Run test/lint/type/build checks.
3. Tag release.
4. Publish package artifacts.

### Public repository hygiene

Keep issue templates, contribution guide, and code-of-conduct up to date.

## 9. Common Mistakes to Avoid

1. Data leakage from preprocessing full dataset before split.
2. Artifact formats without schema version fields.
3. Hardcoding local paths in docs/examples.
4. Coupling pipeline logic directly to one model framework.
5. Missing tests for edge-case labels and empty data.

## 10. Suggested Next Enhancements

1. Hyperparameter tuning module.
2. Cross-validation runner with report aggregation.
3. Plugin architecture for project-specific classifier registries.
4. Experiment tracking integration via optional adapters.
