from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseClassifier(ABC):
    """Common interface that every classifier implementation must satisfy."""

    @abstractmethod
    def fit(self, features: list[list[float]], labels: list[str]) -> None:
        """Train the classifier from feature vectors and labels."""

    @abstractmethod
    def predict(self, features: list[list[float]]) -> list[str]:
        """Return label predictions for each feature vector."""

    @abstractmethod
    def to_artifact(self) -> dict[str, Any]:
        """Serialize model state into a JSON-compatible dictionary."""

    @classmethod
    @abstractmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "BaseClassifier":
        """Restore classifier state from a serialized artifact."""


class BaseDataLoader(ABC):
    """Data source contract for tabular classification workloads."""

    @abstractmethod
    def load(self) -> tuple[list[list[float]], list[str]]:
        """Load and return features and labels."""


class BaseEvaluator(ABC):
    """Evaluation strategy contract used by the training pipeline."""

    @abstractmethod
    def compute(self, actual: list[str], predicted: list[str]) -> dict[str, float]:
        """Compute evaluation metrics from actual and predicted labels."""
