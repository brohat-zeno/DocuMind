"""Logging configuration."""

import logging


def configure_logging() -> None:
    """Enable concise request and application logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
