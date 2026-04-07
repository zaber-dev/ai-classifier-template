from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_classifier.core.base import BaseClassifier
from ai_classifier.core.exceptions import DataFormatError


def _safe_import_sklearn() -> Any:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for model.kind='sklearn'. Install with: pip install .[ml]"
        ) from exc
    return {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
    }


@dataclass(slots=True)
class SklearnClassifierAdapter(BaseClassifier):
    """Adapter that wraps selected sklearn classifiers behind the base interface."""

    algorithm: str = "logistic_regression"
    params: dict[str, Any] = field(default_factory=dict)
    estimator: Any | None = None

    def _ensure_estimator(self) -> None:
        if self.estimator is not None:
            return
        registry = _safe_import_sklearn()
        if self.algorithm not in registry:
            raise DataFormatError(f"Unsupported sklearn algorithm: {self.algorithm}")
        self.estimator = registry[self.algorithm](**self.params)

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        self._ensure_estimator()
        assert self.estimator is not None
        self.estimator.fit(features, labels)

    def predict(self, features: list[list[float]]) -> list[str]:
        self._ensure_estimator()
        assert self.estimator is not None
        return list(self.estimator.predict(features))

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": "sklearn",
            "class_name": self.__class__.__name__,
            "algorithm": self.algorithm,
            "params": self.params,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "SklearnClassifierAdapter":
        return cls(
            algorithm=str(artifact.get("algorithm", "logistic_regression")),
            params=dict(artifact.get("params", {})),
        )
