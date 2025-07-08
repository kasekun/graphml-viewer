"""
Configuration settings for GraphML Viewer application.

This module provides centralized configuration management using environment
variables with sensible defaults and type safety.
"""

import os
from enum import Enum
from pathlib import Path


class LogLevel(str, Enum):
    """Logging levels."""

    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class LayoutAlgorithm(str, Enum):
    """Supported graph layout algorithms."""

    SPRING = "spring"
    CIRCULAR = "circular"
    RANDOM = "random"
    KAMADA_KAWAI = "kamada_kawai"
    SPECTRAL = "spectral"


class VisualizationBackend(str, Enum):
    """Supported visualization backends."""

    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    BOKEH = "bokeh"
    PYVIS = "pyvis"


class AppSettings:
    """Main application settings with environment variable support."""

    def __init__(self) -> None:
        """Initialize settings from environment variables with defaults."""
        # Application metadata
        self.app_name: str = os.getenv("APP_NAME", "GraphML Viewer")
        self.app_version: str = os.getenv("APP_VERSION", "0.1.0")
        self.debug: bool = self._get_bool_env("DEBUG", False)

        # Logging configuration
        self.log_level: LogLevel = LogLevel(os.getenv("LOG_LEVEL", LogLevel.INFO))
        self.log_format: str = os.getenv(
            "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_file: Path | None = self._get_path_env("LOG_FILE")

        # Performance settings
        self.max_nodes_for_centrality: int = self._get_int_env(
            "MAX_NODES_FOR_CENTRALITY", 1000
        )
        self.max_nodes_for_interactive: int = self._get_int_env(
            "MAX_NODES_FOR_INTERACTIVE", 500
        )
        self.use_sampling: bool = self._get_bool_env("USE_SAMPLING", True)

        # Visualization defaults
        self.default_layout: LayoutAlgorithm = LayoutAlgorithm(
            os.getenv("DEFAULT_LAYOUT", LayoutAlgorithm.SPRING)
        )
        self.default_backend: VisualizationBackend = VisualizationBackend(
            os.getenv("DEFAULT_BACKEND", VisualizationBackend.PLOTLY)
        )
        self.default_width: int = self._get_int_env("DEFAULT_WIDTH", 1200)
        self.default_height: int = self._get_int_env("DEFAULT_HEIGHT", 800)
        self.default_node_size: float = self._get_float_env("DEFAULT_NODE_SIZE", 10.0)
        self.default_edge_width: float = self._get_float_env("DEFAULT_EDGE_WIDTH", 1.0)

        # Color schemes
        self.entity_type_colors: dict[str, str] = self._get_dict_env(
            "ENTITY_TYPE_COLORS",
            {
                "person": "#FF6B6B",
                "organization": "#4ECDC4",
                "location": "#45B7D1",
                "event": "#96CEB4",
                "concept": "#FFEAA7",
                "category": "#DDA0DD",
                "geo": "#98D8C8",
                "unknown": "#BDC3C7",
            },
        )

        # File paths
        cache_dir_env = self._get_path_env("CACHE_DIR")
        self.cache_dir: Path = cache_dir_env or (
            Path.home() / ".cache" / "graphml-viewer"
        )

        temp_dir_env = self._get_path_env("TEMP_DIR")
        self.temp_dir: Path = temp_dir_env or (Path.cwd() / "tmp")

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Streamlit configuration
        self.streamlit_host: str = os.getenv("STREAMLIT_HOST", "localhost")
        self.streamlit_port: int = self._get_int_env("STREAMLIT_PORT", 8501)
        self.streamlit_debug: bool = self._get_bool_env("STREAMLIT_DEBUG", False)

        # Analysis settings
        self.community_detection_algorithm: str = os.getenv(
            "COMMUNITY_DETECTION_ALGORITHM", "greedy"
        )
        self.centrality_algorithms: list[str] = self._get_list_env(
            "CENTRALITY_ALGORITHMS", ["degree", "betweenness", "closeness", "pagerank"]
        )

    def _get_bool_env(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, "").lower()
        return value in ("true", "1", "yes", "on") if value else default

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default

    def _get_float_env(self, key: str, default: float) -> float:
        """Get float value from environment variable."""
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            return default

    def _get_path_env(self, key: str, default: Path | None = None) -> Path | None:
        """Get Path value from environment variable."""
        value = os.getenv(key)
        if value:
            return Path(value)
        return default

    def _get_list_env(self, key: str, default: list[str]) -> list[str]:
        """Get list value from environment variable (comma-separated)."""
        value = os.getenv(key)
        if value:
            return [item.strip() for item in value.split(",")]
        return default

    def _get_dict_env(self, key: str, default: dict[str, str]) -> dict[str, str]:
        """Get dictionary value from environment variable (JSON format)."""
        value = os.getenv(key)
        if value:
            try:
                import json

                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
        return default


class DatabaseSettings:
    """Database configuration (for future extensions)."""

    def __init__(self) -> None:
        """Initialize database settings from environment variables."""
        self.db_url: str | None = os.getenv("DATABASE_URL")
        self.db_timeout: int = self._get_int_env("DB_TIMEOUT", 30)
        self.db_pool_size: int = self._get_int_env("DB_POOL_SIZE", 10)

    def _get_int_env(self, key: str, default: int) -> int:
        """Get integer value from environment variable."""
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            return default


def get_settings() -> AppSettings:
    """
    Get application settings instance.

    This function creates and caches the settings instance to ensure
    configuration is loaded only once.

    Returns:
        AppSettings: Configured application settings
    """
    if not hasattr(get_settings, "_settings"):
        get_settings._settings = AppSettings()
    return get_settings._settings


def get_database_settings() -> DatabaseSettings:
    """
    Get database settings instance.

    Returns:
        DatabaseSettings: Configured database settings
    """
    if not hasattr(get_database_settings, "_settings"):
        get_database_settings._settings = DatabaseSettings()
    return get_database_settings._settings


# Global settings instance
settings = get_settings()
