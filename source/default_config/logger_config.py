# logger_config.py
import logging
from logging.handlers import RotatingFileHandler
import os

LOG_DIR = "logs"
LOG_FILE = "app.log"

def setup_logger(name: str) -> logging.Logger:
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # file Handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, LOG_FILE),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3              # keep 3 backup files
    )
    file_handler.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # add handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
