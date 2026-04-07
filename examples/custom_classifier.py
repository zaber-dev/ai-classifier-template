from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_classifier.core.base import BaseClassifier


@dataclass(slots=True)
class ThresholdClassifier(BaseClassifier):
    """Minimal example custom classifier used for extension tutorials."""

    threshold: float = 0.0

    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        # Example classifier is stateless and uses configured threshold.
        return

    def predict(self, features: list[list[float]]) -> list[str]:
        predictions: list[str] = []
        for row in features:
            score = sum(row)
            predictions.append("positive" if score >= self.threshold else "negative")
        return predictions

    def to_artifact(self) -> dict[str, Any]:
        return {
            "model_type": "custom_threshold",
            "state": {"threshold": self.threshold},
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "ThresholdClassifier":
        state = artifact.get("state", {})
        return cls(threshold=float(state.get("threshold", 0.0)))


if __name__ == "__main__":
    demo = ThresholdClassifier(threshold=0.1)
    print(demo.predict([[0.2, 0.1], [-0.4, -0.2]]))
