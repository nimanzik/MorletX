from __future__ import annotations

import logging
from typing import Literal

LoggingLevel = Literal["debug", "info", "warning", "error", "critical"]


def set_log_level(logger: logging.Logger, level: LoggingLevel) -> None:
    """Set the logging level of the logger."""
    level = getattr(logging, level.upper())
    logger.setLevel(level)


def get_logger(name: str, level: LoggingLevel = "info") -> logging.Logger:
    """Get a logger with the specified name and level."""
    try:
        from rich.logging import RichHandler

        handlers = [RichHandler(rich_tracebacks=True)]
        format_ = "%(name)-15s - %(message)s"
    except ImportError:
        handlers = None
        format_ = "%(levelname)-8s - %(name)-15s - %(message)s"

    logger = logging.getLogger(name=name)
    logging.basicConfig(
        level=level.upper(),
        handlers=handlers,
        format=format_,
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )
    return logger
