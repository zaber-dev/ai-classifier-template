from __future__ import annotations

from typing import Protocol


class PipelineCallback(Protocol):
    """Typed callback hooks for lifecycle events in the training pipeline."""

    def on_train_start(self) -> None:
        ...

    def on_train_complete(self, report: dict[str, float]) -> None:
        ...


class NoopCallback:
    """Default callback implementation used when no hooks are provided."""

    def on_train_start(self) -> None:
        return

    def on_train_complete(self, report: dict[str, float]) -> None:
        return
