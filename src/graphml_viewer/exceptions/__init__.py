"""
Custom exceptions for GraphML Viewer with rich context capture.

This module provides specialized exceptions that capture detailed context
information for better debugging and error reporting.
"""

from typing import Any, Dict, Optional


class GraphMLViewerError(Exception):
    """Base exception for all GraphML Viewer errors."""

    def __init__(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize GraphML Viewer error with context.

        Args:
            message: Human-readable error message
            context: Additional context information for debugging
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.original_error = original_error

    def add_context(self, key: str, value: Any) -> "GraphMLViewerError":
        """
        Add context information to the error.

        Args:
            key: Context key
            value: Context value

        Returns:
            Self for method chaining
        """
        self.context[key] = value
        return self

    def get_full_message(self) -> str:
        """
        Get the full error message with context information.

        Returns:
            Complete error message with context
        """
        parts = [self.message]

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.original_error:
            parts.append(f"Original error: {self.original_error}")

        return " | ".join(parts)


class GraphLoadingError(GraphMLViewerError):
    """Error occurred while loading GraphML file."""

    def __init__(
        self,
        file_path: str,
        message: str = "Failed to load GraphML file",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("file_path", file_path)


class GraphProcessingError(GraphMLViewerError):
    """Error occurred while processing graph data."""

    def __init__(
        self,
        operation: str,
        message: str = "Graph processing failed",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("operation", operation)


class VisualizationError(GraphMLViewerError):
    """Error occurred during graph visualization."""

    def __init__(
        self,
        backend: str,
        message: str = "Visualization failed",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("backend", backend)


class AnalysisError(GraphMLViewerError):
    """Error occurred during graph analysis."""

    def __init__(
        self,
        analysis_type: str,
        message: str = "Graph analysis failed",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("analysis_type", analysis_type)


class ConfigurationError(GraphMLViewerError):
    """Error in application configuration."""

    def __init__(
        self,
        config_key: str,
        message: str = "Configuration error",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("config_key", config_key)


class ValidationError(GraphMLViewerError):
    """Error in data validation."""

    def __init__(
        self,
        field_name: str,
        field_value: Any,
        message: str = "Validation failed",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("field_name", field_name)
        self.add_context("field_value", field_value)


class ResourceError(GraphMLViewerError):
    """Error related to system resources (memory, disk, etc.)."""

    def __init__(
        self,
        resource_type: str,
        message: str = "Resource error",
        context: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message, context, original_error)
        self.add_context("resource_type", resource_type)


# Export all exceptions
__all__ = [
    "GraphMLViewerError",
    "GraphLoadingError",
    "GraphProcessingError",
    "VisualizationError",
    "AnalysisError",
    "ConfigurationError",
    "ValidationError",
    "ResourceError",
]
