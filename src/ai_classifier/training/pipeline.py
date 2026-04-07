from __future__ import annotations

from dataclasses import dataclass, field

from ai_classifier.classifiers.sklearn_classifier import SklearnClassifierAdapter
from ai_classifier.classifiers.template_classifier import TemplateCentroidClassifier
from ai_classifier.core.base import BaseClassifier
from ai_classifier.core.config import PipelineConfig
from ai_classifier.data.loaders import CSVClassificationLoader
from ai_classifier.data.preprocessing import split_dataset
from ai_classifier.training.callbacks import NoopCallback, PipelineCallback
from ai_classifier.training.metrics import REGISTRY
from ai_classifier.utils.serialization import save_model


def _build_classifier(kind: str, params: dict[str, object]) -> BaseClassifier:
    if kind == "template":
        return TemplateCentroidClassifier()
    if kind == "sklearn":
        algorithm = str(params.get("algorithm", "logistic_regression"))
        model_params = dict(params.get("model_params", {}))
        return SklearnClassifierAdapter(algorithm=algorithm, params=model_params)
    raise ValueError(f"Unsupported model kind: {kind}")


@dataclass(slots=True)
class TrainingPipeline:
    """Orchestrates local model training, evaluation, and artifact generation."""

    config: PipelineConfig
    callback: PipelineCallback = field(default_factory=NoopCallback)

    def run(self) -> dict[str, float]:
        loader = CSVClassificationLoader(
            path=self.config.dataset.path,
            label_column=self.config.dataset.label_column,
        )
        features, labels = loader.load()
        train_x, train_y, test_x, test_y = split_dataset(
            features=features,
            labels=labels,
            test_size=self.config.dataset.test_size,
            random_seed=self.config.dataset.random_seed,
            shuffle=self.config.dataset.shuffle,
        )

        classifier = _build_classifier(self.config.model.kind, self.config.model.params)
        self.callback.on_train_start()
        classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)

        report: dict[str, float] = {}
        for metric in self.config.training.metrics:
            report[metric] = round(REGISTRY[metric](test_y, predictions), 6)

        save_model(classifier, self.config.output.model_path)
        save_model_report(
            path=self.config.output.report_path,
            report=report,
            model_kind=self.config.model.kind,
            train_size=len(train_x),
            test_size=len(test_x),
        )
        self.callback.on_train_complete(report)
        return report


def save_model_report(
    path: str,
    report: dict[str, float],
    model_kind: str,
    train_size: int,
    test_size: int,
) -> None:
    from json import dumps
    from pathlib import Path

    payload = {
        "model_kind": model_kind,
        "train_size": train_size,
        "test_size": test_size,
        "metrics": report,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps(payload, indent=2), encoding="utf-8")
