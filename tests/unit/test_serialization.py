from pathlib import Path

from ai_classifier.classifiers.template_classifier import TemplateCentroidClassifier
from ai_classifier.utils.serialization import load_model, save_model


def test_template_model_roundtrip(tmp_path: Path) -> None:
    model = TemplateCentroidClassifier()
    model.fit([[1.0, 1.0], [-1.0, -1.0]], ["positive", "negative"])

    artifact = tmp_path / "model.json"
    save_model(model, str(artifact))
    loaded = load_model(str(artifact))

    assert loaded.predict([[0.8, 0.7], [-0.9, -0.8]]) == ["positive", "negative"]
