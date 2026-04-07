from __future__ import annotations

import csv
from pathlib import Path

from ai_classifier.core.base import BaseDataLoader
from ai_classifier.core.exceptions import DataFormatError


class CSVClassificationLoader(BaseDataLoader):
    """Loads numeric feature columns and a target label from a CSV file."""

    def __init__(self, path: str, label_column: str) -> None:
        self.path = path
        self.label_column = label_column

    def load(self) -> tuple[list[list[float]], list[str]]:
        csv_path = Path(self.path)
        if not csv_path.exists():
            raise DataFormatError(f"CSV file not found: {self.path}")

        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DataFormatError("CSV file is missing a header row")
            if self.label_column not in reader.fieldnames:
                raise DataFormatError(
                    f"CSV is missing required label column: {self.label_column}"
                )

            feature_columns = [name for name in reader.fieldnames if name != self.label_column]
            if not feature_columns:
                raise DataFormatError("CSV must include at least one feature column")

            features: list[list[float]] = []
            labels: list[str] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    feature_row = [float(row[column]) for column in feature_columns]
                except (TypeError, ValueError) as exc:
                    raise DataFormatError(
                        f"Row {row_number} contains non-numeric feature values"
                    ) from exc

                label = row.get(self.label_column)
                if label is None or label == "":
                    raise DataFormatError(f"Row {row_number} has an empty label")

                features.append(feature_row)
                labels.append(label)

        if not features:
            raise DataFormatError("CSV contains no data rows")
        return features, labels
