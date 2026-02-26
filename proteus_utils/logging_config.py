"""
logging_config.py
Centralized logging configuration for the Proteus pipeline.

Usage:
    from proteus_utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing started...")
    logger.debug("Detailed debug info")
    logger.warning("Something unexpected")
"""

import logging
import sys
from typing import Optional


# Module-level logger cache
_loggers = {}

# Default format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
SIMPLE_FORMAT = '%(levelname)s: %(message)s'
COMPACT_FORMAT = '%(message)s'


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_style: str = 'simple'
) -> None:
    """
    Configure the root logger for the Proteus pipeline.
    
    Call this once at the start of the main script to configure logging.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        log_file: Optional path to write logs to file
        format_style: 'default' (with timestamps), 'simple' (level + message), 
                      or 'compact' (message only)
    """
    format_map = {
        'default': DEFAULT_FORMAT,
        'simple': SIMPLE_FORMAT,
        'compact': COMPACT_FORMAT,
    }
    log_format = format_map.get(format_style, SIMPLE_FORMAT)
    
    # Configure root logger
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger for proteus modules
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set level for proteus modules
    for logger_name in ['proteus_utils', 'models', 'config', 'trainer', 'analysis']:
        logging.getLogger(logger_name).setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    return _loggers[name]


def set_verbosity(verbose: bool = True) -> None:
    """
    Set logging verbosity for interactive use.
    
    Args:
        verbose: If True, set DEBUG level; if False, set WARNING level
    """
    level = logging.DEBUG if verbose else logging.WARNING
    setup_logging(level=level)
