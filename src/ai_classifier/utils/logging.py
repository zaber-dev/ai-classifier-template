from __future__ import annotations

import logging


def get_logger(name: str = "ai_classifier") -> logging.Logger:
    """Create or fetch a console logger with predictable formatting."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger
