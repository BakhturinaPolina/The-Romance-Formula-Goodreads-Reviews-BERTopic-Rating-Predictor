"""
Logging configuration for Romance Novel NLP Research Project.

This module sets up comprehensive logging for all components of the project,
including file rotation, formatting, and different log levels for different
components.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any

from .settings import LOGS_DIR, LOGGING


def setup_logging(
    name: str = "romance_research",
    level: str = None,
    log_file: Path = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console_output: Whether to output to console
        file_output: Whether to output to file
        
    Returns:
        Configured logger instance
    """
    
    # Use default settings if not provided
    level = level or LOGGING["level"]
    log_file = log_file or LOGGING["file"]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt=LOGGING["format"],
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(levelname)s - %(message)s"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=LOGGING["backup_count"],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for a specific component.
    
    Args:
        name: Component name (e.g., 'data_processing', 'nlp_analysis')
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"romance_research.{name}")
    return logging.getLogger("romance_research")


def setup_component_logging(component_name: str) -> logging.Logger:
    """
    Set up logging for a specific component with appropriate settings.
    
    Args:
        component_name: Name of the component (e.g., 'data_exploration')
        
    Returns:
        Configured logger for the component
    """
    
    # Create component-specific log file
    component_log_file = LOGS_DIR / f"{component_name}.log"
    
    # Set up logging for this component
    logger = setup_logging(
        name=f"romance_research.{component_name}",
        log_file=component_log_file,
        console_output=True,
        file_output=True
    )
    
    return logger


def log_function_call(logger: logging.Logger = None):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if logger is None:
                func_logger = get_logger(func.__module__)
            else:
                func_logger = logger
                
            func_logger.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.info(f"Function {func.__name__} completed successfully")
                return result
            except Exception as e:
                func_logger.error(f"Function {func.__name__} failed with error: {str(e)}")
                raise
                
        return wrapper
    return decorator


def log_data_info(logger: logging.Logger, data_name: str, data_info: Dict[str, Any]):
    """
    Log information about a dataset.
    
    Args:
        logger: Logger instance
        data_name: Name of the dataset
        data_info: Dictionary with dataset information
    """
    logger.info(f"Dataset: {data_name}")
    for key, value in data_info.items():
        logger.info(f"  {key}: {value}")


def log_processing_step(logger: logging.Logger, step_name: str, step_info: Dict[str, Any] = None):
    """
    Log information about a processing step.
    
    Args:
        logger: Logger instance
        step_name: Name of the processing step
        step_info: Additional information about the step
    """
    logger.info(f"Processing step: {step_name}")
    if step_info:
        for key, value in step_info.items():
            logger.info(f"  {key}: {value}")


def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
    """
    Log an error with additional context information.
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error details: {error}")


# Initialize main logger
main_logger = setup_logging()

# Component-specific loggers
data_exploration_logger = setup_component_logging("data_exploration")
data_processing_logger = setup_component_logging("data_processing")
nlp_analysis_logger = setup_component_logging("nlp_analysis")
utils_logger = setup_component_logging("utils")
