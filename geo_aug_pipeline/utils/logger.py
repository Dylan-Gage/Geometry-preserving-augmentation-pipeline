"""
utils/logger.py
Centralised logging setup. Call get_logger(name) from any module.
"""
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_to_file: bool = False,
                  log_filename: str = "pipeline.log", logs_dir: str = "logs") -> None:
    """Configure root logger. Call once from main.py."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        Path(logs_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(logs_dir) / log_filename, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        handlers.append(fh)

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, handlers=handlers)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
