from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ai_classifier.core.config import PipelineConfig
from ai_classifier.training.pipeline import TrainingPipeline
from ai_classifier.utils.serialization import load_model


def _predict_csv(model_path: str, input_path: str, output_path: str) -> None:
    model = load_model(model_path)
    source = Path(input_path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    features: list[list[float]] = []
    for row in rows:
        vector: list[float] = []
        for key in fieldnames:
            vector.append(float(row[key]))
        features.append(vector)

    predictions = model.predict(features)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction"])
        for prediction in predictions:
            writer.writerow([prediction])


def _evaluate(model_path: str, input_path: str, label_column: str, output_path: str) -> None:
    model = load_model(model_path)
    with Path(input_path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        feature_columns = [name for name in fieldnames if name != label_column]
        rows = list(reader)

    features = [[float(row[column]) for column in feature_columns] for row in rows]
    actual = [str(row[label_column]) for row in rows]
    predicted = model.predict(features)
    correct = sum(1 for left, right in zip(actual, predicted) if left == right)
    report = {
        "accuracy": (correct / len(actual)) if actual else 0.0,
        "records": len(actual),
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-classifier",
        description="Local-first classifier toolkit template",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model from a YAML config")
    train_parser.add_argument("--config", required=True, help="Path to YAML config")

    predict_parser = subparsers.add_parser("predict", help="Run model predictions on CSV")
    predict_parser.add_argument("--model", required=True, help="Model artifact path")
    predict_parser.add_argument("--input", required=True, help="Input CSV with feature columns")
    predict_parser.add_argument("--output", required=True, help="Output CSV for predictions")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on labeled CSV")
    eval_parser.add_argument("--model", required=True, help="Model artifact path")
    eval_parser.add_argument("--input", required=True, help="Input CSV with labels")
    eval_parser.add_argument("--label-column", required=True, help="Name of label column")
    eval_parser.add_argument("--output", required=True, help="Output JSON report")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        config = PipelineConfig.from_yaml(args.config)
        pipeline = TrainingPipeline(config=config)
        report = pipeline.run()
        print(json.dumps(report, indent=2))
        return

    if args.command == "predict":
        _predict_csv(args.model, args.input, args.output)
        print(f"Predictions written to {args.output}")
        return

    if args.command == "evaluate":
        _evaluate(args.model, args.input, args.label_column, args.output)
        print(f"Evaluation report written to {args.output}")
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
