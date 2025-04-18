"""
Logger utility for Test Runner.
"""

import logging
import sys
from pathlib import Path


def configure_process_logging(verbose: bool) -> logging.Logger:
    """Configure logging for a worker process.
    
    Args:
        verbose (bool): Whether to enable verbose logging. When disabled, no logs will be shown.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger('test_runner')
    logger.setLevel(logging.INFO if verbose else logging.CRITICAL)  # Only show logs in verbose mode
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add a new handler that only shows logs if verbose is True
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO if verbose else logging.CRITICAL)  # Only show logs in verbose mode
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def setup_logger(log_file=None, verbose=False):
    """
    Set up and configure the logger.
    
    Args:
        log_file (str, optional): Path to the log file. If None, logs to console only.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("test_runner")
    logger.setLevel(logging.DEBUG if verbose else logging.CRITICAL)  # Only show logs in verbose mode
    
    # Create formatter
    formatter = logging.Formatter(
        '%(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.CRITICAL)  # Only show logs in verbose mode
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 