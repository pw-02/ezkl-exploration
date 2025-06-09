import logging
import sys
def setup_logger(name, log_file=None, level=logging.INFO):
    """Set up a logger that logs to both console and file (if given)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  
    # Prevent duplicated logs if called multiple times
    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
