from ai_classifier.classifiers.template_classifier import TemplateCentroidClassifier


def test_template_classifier_predicts_known_clusters() -> None:
    model = TemplateCentroidClassifier()
    features = [[1.0, 1.0], [1.2, 0.8], [-1.0, -1.0], [-0.8, -1.1]]
    labels = ["positive", "positive", "negative", "negative"]
    model.fit(features, labels)

    predictions = model.predict([[1.1, 0.9], [-0.9, -1.0]])
    assert predictions == ["positive", "negative"]
