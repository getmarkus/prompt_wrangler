"""Logging configuration for the prompt wrangler."""

import sys
from typing import Optional

from loguru import logger


def setup_logger(log_level: str = "INFO") -> None:
    """Configure logger for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    # Remove default handlers
    logger.remove()
    
    # Add console handler with formatting
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level=log_level,
        colorize=True,
    )
    
    logger.debug("Logger initialized with level: {}", log_level)


# Initialize logger with default settings
setup_logger()
