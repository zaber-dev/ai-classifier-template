from pathlib import Path

import pytest

from ai_classifier.core.config import PipelineConfig
from ai_classifier.core.exceptions import ConfigurationError


def test_pipeline_config_from_dict_validates_dataset_path(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("f1,label\n1.0,a\n", encoding="utf-8")

    config = PipelineConfig.from_dict(
        {
            "dataset": {
                "path": str(data),
                "label_column": "label",
                "test_size": 0.3,
            }
        }
    )
    assert config.dataset.path == str(data)


def test_pipeline_config_rejects_invalid_test_size(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("f1,label\n1.0,a\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        PipelineConfig.from_dict(
            {
                "dataset": {
                    "path": str(data),
                    "label_column": "label",
                    "test_size": 1.2,
                }
            }
        )
