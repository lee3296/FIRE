"""Narrow compatibility helpers that do not change model numerics."""
from __future__ import annotations

import logging


class _GenerationLengthWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        # max_new_tokens already wins. Suppress only this redundant diagnostic;
        # do not alter generation settings or hide other Transformers warnings.
        return not (
            "Both `max_new_tokens`" in message
            and "`max_length`" in message
            and "`max_new_tokens` will take precedence" in message
        )


def configure_transformers_logging() -> None:
    logger = logging.getLogger("transformers.generation.utils")
    if not any(isinstance(item, _GenerationLengthWarningFilter) for item in logger.filters):
        logger.addFilter(_GenerationLengthWarningFilter())
