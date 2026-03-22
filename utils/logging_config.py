"""Centralized logging configuration for the RAG system.

Provides:
- Configurable log levels via environment variable
- Colored console output for development
- File logging for production debugging
- Consistent logger format across modules
"""
import logging
import os
import sys
from pathlib import Path

# Log level from environment (default: INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "")

# Colors for console output
class Colors:
    DEBUG = "\033[36m"    # Cyan
    INFO = "\033[92m"     # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"    # Red
    CRITICAL = "\033[95m" # Magenta
    RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DEBUG,
        logging.INFO: Colors.INFO,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.CRITICAL,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"
        return super().format(record)


def setup_logging(
    level: str = LOG_LEVEL,
    log_file: str | None = LOG_FILE or None,
    console: bool = True,
) -> None:
    """Configure logging for the entire application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console: Enable console output
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Remove existing handlers
    logger.handlers.clear()

    # Common format
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler with colors
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter(fmt, datefmt))
        logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Usage:
        from utils.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Message")

    Args:
        name: Logger name (use __name__ for module-level loggers)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Auto-setup on import
setup_logging()
