"""
Logging Configuration Helper

This module provides centralized logging configuration for the corpus creation pipeline.
It sets up both file and console logging with appropriate levels and formatting.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def configure_logging(
    log_level: str = "DEBUG",
    log_dir: Optional[Path] = None,
    log_filename: str = "pipeline_debug.log",
    console_level: str = "INFO"
) -> None:
    """
    Configure logging for the corpus creation pipeline.
    
    Args:
        log_level: Logging level for file output (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (defaults to logs/ in project root)
        log_filename: Name of the main log file
        console_level: Logging level for console output
    """
    # Create logs directory
    if log_dir is None:
        # Find project root (look for .git or src directory)
        current_dir = Path(__file__).parent
        project_root = current_dir
        while project_root.parent != project_root:
            if (project_root / '.git').exists() or (project_root / 'src').exists():
                break
            project_root = project_root.parent
        
        log_dir = project_root / 'logs'
    
    log_dir.mkdir(exist_ok=True)
    
    # Convert string levels to logging constants
    file_level = getattr(logging, log_level.upper(), logging.DEBUG)
    console_level_num = getattr(logging, console_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to most verbose level
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler with rotation
    log_file = log_dir / log_filename
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level_num)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - File: {log_file} (level: {log_level}), Console: {console_level}")
    logger.debug("Debug logging enabled - detailed execution information will be recorded")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the configured settings.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log a function call with its parameters.
    
    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_function_result(logger: logging.Logger, func_name: str, result, **kwargs):
    """
    Log a function result.
    
    Args:
        logger: Logger instance
        func_name: Name of the function that was called
        result: Function result to log
        **kwargs: Additional context to log
    """
    context = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"{func_name} returned: {result} (context: {context})")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log an error with full context.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about where the error occurred
    """
    logger.error(f"Error in {context}: {type(error).__name__}: {error}")
    logger.debug(f"Full traceback:", exc_info=True)
