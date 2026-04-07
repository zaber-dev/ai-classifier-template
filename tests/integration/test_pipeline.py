from pathlib import Path

from ai_classifier.core.config import PipelineConfig
from ai_classifier.training.pipeline import TrainingPipeline


def test_training_pipeline_generates_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "train.csv"
    data_path.write_text(
        "f1,f2,label\n"
        "1.0,1.0,yes\n"
        "1.1,0.9,yes\n"
        "-1.0,-1.0,no\n"
        "-0.9,-1.1,no\n",
        encoding="utf-8",
    )

    model_path = tmp_path / "model.json"
    report_path = tmp_path / "report.json"

    config = PipelineConfig.from_dict(
        {
            "dataset": {
                "path": str(data_path),
                "label_column": "label",
                "test_size": 0.25,
                "shuffle": True,
                "random_seed": 3,
            },
            "model": {"kind": "template", "params": {}},
            "training": {"metrics": ["accuracy", "f1"]},
            "output": {
                "model_path": str(model_path),
                "report_path": str(report_path),
            },
        }
    )

    report = TrainingPipeline(config=config).run()
    assert "accuracy" in report
    assert model_path.exists()
    assert report_path.exists()
