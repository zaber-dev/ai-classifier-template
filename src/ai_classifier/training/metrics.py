from __future__ import annotations


def accuracy(actual: list[str], predicted: list[str]) -> float:
    correct = sum(1 for left, right in zip(actual, predicted) if left == right)
    return correct / len(actual) if actual else 0.0


def precision(actual: list[str], predicted: list[str]) -> float:
    labels = sorted(set(actual) | set(predicted))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fp = sum(1 for a, p in zip(actual, predicted) if a != label and p == label)
        scores.append(tp / (tp + fp) if (tp + fp) else 0.0)
    return sum(scores) / len(scores)


def recall(actual: list[str], predicted: list[str]) -> float:
    labels = sorted(set(actual) | set(predicted))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for a, p in zip(actual, predicted) if a == label and p == label)
        fn = sum(1 for a, p in zip(actual, predicted) if a == label and p != label)
        scores.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return sum(scores) / len(scores)


def f1(actual: list[str], predicted: list[str]) -> float:
    p = precision(actual, predicted)
    r = recall(actual, predicted)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


REGISTRY = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}
