import logging
import warnings

warnings.filterwarnings("ignore")


# Logging configuration
def configure_logging(log_file=None):
    log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)

    # Create a logger
    logger = logging.getLogger(__name__)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        logger.addHandler(file_handler)

    return logger


logger = configure_logging()
