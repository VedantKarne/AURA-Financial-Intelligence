import logging
import sys

def setup_logger(name: str = "AURA", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a centralized logger for the Financial Earnings Intelligence Platform.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger is already initialized
    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False
        
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

# Create a default global logger instance
logger = setup_logger()
