from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any

from ai_classifier.classifiers.sklearn_classifier import SklearnClassifierAdapter
from ai_classifier.classifiers.template_classifier import TemplateCentroidClassifier
from ai_classifier.core.base import BaseClassifier
from ai_classifier.core.exceptions import ModelArtifactError


def save_model(model: BaseClassifier, path: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    artifact = model.to_artifact()
    if artifact.get("model_type") == "sklearn" and destination.suffix == ".joblib":
        _save_sklearn_joblib(model, artifact, destination)
        return

    destination.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def load_model(path: str) -> BaseClassifier:
    source = Path(path)
    if not source.exists():
        raise ModelArtifactError(f"Model artifact not found: {path}")

    if source.suffix == ".joblib":
        return _load_sklearn_joblib(source)

    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    model_type = payload.get("model_type")
    if model_type == "template":
        return TemplateCentroidClassifier.from_artifact(payload)
    if model_type == "sklearn":
        return SklearnClassifierAdapter.from_artifact(payload)
    raise ModelArtifactError(f"Unknown model_type in artifact: {model_type}")


def _save_sklearn_joblib(
    model: BaseClassifier,
    artifact_metadata: dict[str, Any],
    destination: Path,
) -> None:
    try:
        joblib = importlib.import_module("joblib")
    except ImportError as exc:
        raise ModelArtifactError(
            "joblib is required to save sklearn models as .joblib. Install with: pip install .[ml]"
        ) from exc

    estimator = getattr(model, "estimator", None)
    if estimator is None:
        raise ModelArtifactError("sklearn adapter estimator is missing; train before saving")
    payload = {"metadata": artifact_metadata, "estimator": estimator}
    joblib.dump(payload, destination)


def _load_sklearn_joblib(source: Path) -> BaseClassifier:
    try:
        joblib = importlib.import_module("joblib")
    except ImportError as exc:
        raise ModelArtifactError(
            "joblib is required to load .joblib models. Install with: pip install .[ml]"
        ) from exc

    payload = joblib.load(source)
    metadata = payload.get("metadata", {})
    estimator = payload.get("estimator")
    adapter = SklearnClassifierAdapter.from_artifact(metadata)
    adapter.estimator = estimator
    return adapter
