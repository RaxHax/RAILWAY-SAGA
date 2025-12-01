"""Application logging configuration."""

import logging
from datetime import datetime
from pathlib import Path


def setup_logging() -> Path:
    """
    Configure application logging with both console + file handlers.
    Generates a unique log file per run under <repo>/logs.
    """
    root_logger = logging.getLogger()

    # Avoid duplicate handler stacking when running with reload
    if getattr(setup_logging, "_configured", False):
        return getattr(setup_logging, "_log_file_path")

    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"media-search-engine_{timestamp}.log"

    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"
    formatter = logging.Formatter(log_format, "%Y-%m-%d %H:%M:%S")

    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    setup_logging._configured = True
    setup_logging._log_file_path = log_file
    return log_file


LOG_FILE_PATH = setup_logging()
logger = logging.getLogger(__name__)
logger.info("File logging active: %s", LOG_FILE_PATH)
