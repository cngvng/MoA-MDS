"""Centralized logging setup using loguru with organized folder structure.

This module provides a flexible logger configuration for research experiments,
supporting:
- Organized log directories by module/experiment name
- Console and file handlers with different log levels
- Automatic file rotation and retention
- Structured logging with context information
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger


class SetupLogger:
    """Centralized logger setup with loguru for research experiments.
    
    Features:
    - Console logging (configurable level)
    - File logging with rotation and retention
    - Organized folder structure (logs/module_name/)
    - Support for different log levels per handler
    - Structured logging with context
    
    Example:
        >>> # Basic usage
        >>> logger = SetupLogger(
        ...     log_level="INFO",
        ...     log_file="src/logs/training/experiment_1.log"
        ... ).get_logger()
        >>> logger.info("Training started")
        
        >>> # Advanced usage with custom settings
        >>> logger = SetupLogger(
        ...     log_level="DEBUG",
        ...     log_file="logs/data_processing/preprocess.log",
        ...     rotation="10 MB",
        ...     retention="7 days",
        ...     enqueue=True
        ... ).get_logger()
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[Union[str, Path]] = None,
        rotation: str = "10 MB",
        retention: str = "10 days",
        enqueue: bool = True,
        format_string: Optional[str] = None
    ):
        """Initialize the logger setup.
        
        Args:
            log_level: Console logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Path to log file. If None, only console logging is enabled.
                     Parent directories will be created automatically.
            rotation: When to rotate log files (e.g., "10 MB", "1 day", "1 week")
            retention: How long to keep old log files (e.g., "7 days", "1 month")
            enqueue: Enable async logging for better performance
            format_string: Custom format string. If None, uses default format.
        """
        self.log_level = log_level.upper()
        self.log_file = Path(log_file) if log_file else None
        self.rotation = rotation
        self.retention = retention
        self.enqueue = enqueue
        
        # Default format with colors for console, structured for file
        self.console_format = format_string or (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        
        self.file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{extra[experiment_id]} | "
            "{message}"
        )
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure loguru logger with console and file handlers."""
        # Remove default handler
        logger.remove()
        logger.configure(extra={"experiment_id": "default"})
        
        # Add console handler with color support
        logger.add(
            sys.stderr,
            format=self.console_format,
            level=self.log_level,
            colorize=True,
            enqueue=self.enqueue
        )
        
        # Add file handler if log_file is specified
        if self.log_file:
            # Create parent directories if they don't exist
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # File handler always logs at DEBUG level for detailed history
            logger.add(
                str(self.log_file),
                format=self.file_format,
                level="DEBUG",
                rotation=self.rotation,
                retention=self.retention,
                compression="zip",
                enqueue=self.enqueue,
            )
            
            logger.info(f"Log file initialized: {self.log_file}")
    
    def get_logger(self):
        """Get the configured logger instance.
        
        Returns:
            Configured loguru logger instance
        """
        return logger
    
    @staticmethod
    def create_experiment_logger(
        experiment_name: str,
        log_dir: str = "logs",
        log_level: str = "INFO"
    ):
        """Create a logger for a specific experiment with organized structure.
        
        Args:
            experiment_name: Name of the experiment (used for folder organization)
            log_dir: Base directory for logs (default: "logs")
            log_level: Console logging level
            
        Returns:
            Configured logger instance
            
        Example:
            >>> logger = SetupLogger.create_experiment_logger(
            ...     experiment_name="bert_finetuning_v1",
            ...     log_dir="logs"
            ... )
            >>> logger.info("Experiment started")
        """
        log_file = Path(log_dir) / experiment_name / f"{experiment_name}.log"
        return SetupLogger(
            log_level=log_level,
            log_file=log_file
        ).get_logger()
    
    @staticmethod
    def create_module_logger(
        module_name: str,
        log_dir: str = "logs",
        log_level: str = "INFO"
    ):
        """Create a logger for a specific module (e.g., data_processing, training).
        
        Args:
            module_name: Name of the module
            log_dir: Base directory for logs
            log_level: Console logging level
            
        Returns:
            Configured logger instance
            
        Example:
            >>> logger = SetupLogger.create_module_logger(
            ...     module_name="data_processing"
            ... )
            >>> logger.debug("Processing batch 1/100")
        """
        log_file = Path(log_dir) / module_name / f"{module_name}.log"
        return SetupLogger(
            log_level=log_level,
            log_file=log_file
        ).get_logger()


# Environment variable support for flexible configuration
def get_logger_from_env(
    module_name: str,
    default_level: str = "INFO"
) -> logger:
    """Get a logger configured from environment variables.
    
    Environment Variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_DIR: Base log directory (default: logs)
        LOG_FILE_PATH: Override complete log file path
    
    Args:
        module_name: Name of the module/experiment
        default_level: Default log level if not set in env
        
    Returns:
        Configured logger instance
    """
    log_level = os.environ.get("LOG_LEVEL", default_level)
    log_dir = os.environ.get("LOG_DIR", "logs")
    
    # Allow complete override of log file path
    log_file_path = os.environ.get("LOG_FILE_PATH")
    if log_file_path:
        log_file = Path(log_file_path)
    else:
        log_file = Path(log_dir) / module_name / f"{module_name}.log"
    
    return SetupLogger(
        log_level=log_level,
        log_file=log_file
    ).get_logger()


# Convenience function for quick setup
def get_logger(
    name: str = "main",
    level: str = "INFO",
    log_to_file: bool = True
) -> logger:
    """Quick logger setup with sensible defaults.
    
    Args:
        name: Logger name (used for file naming)
        level: Log level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    if log_to_file:
        log_file = f"logs/{name}/{name}.log"
    else:
        log_file = None
    
    return SetupLogger(
        log_level=level,
        log_file=log_file
    ).get_logger()
