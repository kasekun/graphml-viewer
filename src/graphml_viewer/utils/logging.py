"""
Comprehensive logging utilities for GraphML Viewer.

This module provides structured logging with context capture, performance
monitoring, and error tracking capabilities.
"""

import functools
import logging
import sys
import time
from pathlib import Path
from typing import Any


# Fallback settings if config not available
class _FallbackSettings:
    log_level = "INFO"
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file = None
    debug = False


settings = _FallbackSettings()


class ContextFilter(logging.Filter):
    """Add context information to log records."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        """
        Initialize context filter.

        Args:
            context: Additional context to add to all log records
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record."""
        for key, value in self.context.items():
            setattr(record, key, value)
        return True


class PerformanceLogger:
    """Logger for tracking performance metrics."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize performance logger."""
        self.logger = logger
        self._timers: dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self._timers[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")

    def end_timer(self, operation: str, log_level: int = logging.INFO) -> float:
        """
        End timing an operation and log the duration.

        Args:
            operation: Name of the operation
            log_level: Logging level for the duration message

        Returns:
            Duration in seconds
        """
        if operation not in self._timers:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0

        duration = time.time() - self._timers[operation]
        del self._timers[operation]

        self.logger.log(
            log_level, f"Operation '{operation}' completed in {duration:.3f}s"
        )
        return duration

    def log_memory_usage(self, operation: str) -> None:
        """Log current memory usage for an operation."""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.logger.info(f"Memory usage for '{operation}': {memory_mb:.2f} MB")
        except ImportError:
            self.logger.debug("psutil not available for memory monitoring")


def setup_logging(
    name: str = "graphml_viewer",
    level: str | None = None,
    log_file: Path | None = None,
    format_string: str | None = None,
    context: dict[str, Any] | None = None,
) -> logging.Logger:
    """
    Set up comprehensive logging for the application.

    Args:
        name: Logger name
        level: Logging level string (uses settings if not provided)
        log_file: Optional log file path (uses settings if not provided)
        format_string: Log format string (uses settings if not provided)
        context: Additional context to add to all log records

    Returns:
        Configured logger instance
    """
    # Use settings as defaults
    level_str = level or getattr(settings, "log_level", "INFO")
    log_file = log_file or getattr(settings, "log_file", None)
    format_string = format_string or getattr(
        settings, "log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Convert string level to numeric
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # Add context filter if provided
    if context:
        context_filter = ContextFilter(context)
        console_handler.addFilter(context_filter)

    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        try:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)

            if context:
                file_handler.addFilter(context_filter)

            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")

        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    return logger


def get_logger(name: str, context: dict[str, Any] | None = None) -> logging.Logger:
    """
    Get a logger instance with optional context.

    Args:
        name: Logger name (typically module name)
        context: Additional context for this logger

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        logger = setup_logging(name, context=context)
    elif context:
        # Add context filter to existing handlers
        context_filter = ContextFilter(context)
        for handler in logger.handlers:
            handler.addFilter(context_filter)

    return logger


def log_function_call(
    logger: logging.Logger | None = None,
    log_level: int = logging.DEBUG,
    log_args: bool = True,
    log_result: bool = False,
    log_duration: bool = True,
):
    """
    Decorator to log function calls with arguments, results, and timing.

    Args:
        logger: Logger to use (creates one if not provided)
        log_level: Logging level for the function call logs
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log function execution time

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            func_logger = logger or get_logger(func.__module__)

            # Log function entry
            func_name = f"{func.__module__}.{func.__name__}"

            if log_args:
                args_str = ", ".join(str(arg) for arg in args)
                kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                func_logger.log(log_level, f"Calling {func_name}({all_args})")
            else:
                func_logger.log(log_level, f"Calling {func_name}")

            # Execute function with timing
            start_time = time.time() if log_duration else None

            try:
                result = func(*args, **kwargs)

                # Log success
                if log_duration and start_time is not None:
                    duration = time.time() - start_time
                    func_logger.log(
                        log_level, f"{func_name} completed in {duration:.3f}s"
                    )

                if log_result:
                    func_logger.log(log_level, f"{func_name} returned: {result}")

                return result

            except Exception as e:
                # Log error with context
                if log_duration and start_time is not None:
                    duration = time.time() - start_time
                    func_logger.error(f"{func_name} failed after {duration:.3f}s: {e}")
                else:
                    func_logger.error(f"{func_name} failed: {e}")
                raise

        return wrapper

    return decorator


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: dict[str, Any] | None = None,
    level: int = logging.ERROR,
) -> None:
    """
    Log an error with rich context information.

    Args:
        logger: Logger to use
        error: Exception to log
        context: Additional context information
        level: Logging level
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        error_info.update(context)

    # Check if it's one of our custom exceptions with get_full_message method
    if hasattr(error, "get_full_message") and callable(error.get_full_message):
        message = error.get_full_message()  # type: ignore
    else:
        message = str(error)

    # Format context for logging
    if error_info:
        context_str = " | ".join(f"{k}: {v}" for k, v in error_info.items())
        full_message = f"{message} | {context_str}"
    else:
        full_message = message

    logger.log(level, full_message, exc_info=True)


# Global logger instance
_default_logger: logging.Logger | None = None


def get_default_logger() -> logging.Logger:
    """Get the default application logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging("graphml_viewer")
    return _default_logger
