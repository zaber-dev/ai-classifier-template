from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from ai_classifier.core.base import BaseClassifier
from ai_classifier.core.exceptions import DataFormatError


def _euclidean(left: list[float], right: list[float]) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


@dataclass(slots=True)
class TemplateCentroidClassifier(BaseClassifier):
    """A simple nearest-centroid classifier for template/demo usage.

    This implementation is intentionally lightweight and local-only so users can
    customize it quickly for project-specific classifiers.
    """

    class_centroids: dict[str, list[float]] = field(default_factory=dict)

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        if not features or not labels:
            raise DataFormatError("features and labels must be non-empty")
        if len(features) != len(labels):
            raise DataFormatError("features and labels must have equal length")

        grouped: dict[str, list[list[float]]] = {}
        for row, label in zip(features, labels):
            grouped.setdefault(label, []).append(row)

        centroids: dict[str, list[float]] = {}
        for label, rows in grouped.items():
            dimensions = len(rows[0])
            centroid = []
            for index in range(dimensions):
                centroid.append(sum(r[index] for r in rows) / len(rows))
            centroids[label] = centroid

        self.class_centroids = centroids

    def predict(self, features: list[list[float]]) -> list[str]:
        if not self.class_centroids:
            raise DataFormatError("classifier must be fit before prediction")

        labels = list(self.class_centroids.keys())
        predictions: list[str] = []
        for row in features:
            best_label = min(labels, key=lambda candidate: _euclidean(row, self.class_centroids[candidate]))
            predictions.append(best_label)
        return predictions

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": "template",
            "class_name": self.__class__.__name__,
            "state": {"class_centroids": self.class_centroids},
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "TemplateCentroidClassifier":
        state = artifact.get("state", {})
        centroids = state.get("class_centroids", {})
        if not isinstance(centroids, dict):
            raise DataFormatError("Invalid template artifact: state.class_centroids must be a dictionary")
        return cls(class_centroids=centroids)
