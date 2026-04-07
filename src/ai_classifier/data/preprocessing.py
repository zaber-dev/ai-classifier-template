from __future__ import annotations

import random


def split_dataset(
    features: list[list[float]],
    labels: list[str],
    test_size: float,
    random_seed: int,
    shuffle: bool,
) -> tuple[list[list[float]], list[str], list[list[float]], list[str]]:
    """Split features and labels into deterministic train/test partitions."""

    indices = list(range(len(features)))
    if shuffle:
        random.Random(random_seed).shuffle(indices)

    cutoff = max(1, int(len(indices) * (1.0 - test_size)))
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    if not test_indices:
        test_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    train_x = [features[index] for index in train_indices]
    train_y = [labels[index] for index in train_indices]
    test_x = [features[index] for index in test_indices]
    test_y = [labels[index] for index in test_indices]
    return train_x, train_y, test_x, test_y
