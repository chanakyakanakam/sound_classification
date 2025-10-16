"""
Path: /home/chanakya/sound_classification/utils/logger.py
Logging Utilities for Pump-Net - Console Only
"""

import logging
import sys
from config.config import settings

def setup_logger(name: str = "pump_net") -> logging.Logger:
    """
    Setup logger with console handler only (no file logging)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT
    )
    
    # Console handler only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logger()