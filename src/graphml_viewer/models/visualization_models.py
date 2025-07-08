"""
Data models for visualization configuration and styling.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LayoutAlgorithm(str, Enum):
    """Supported graph layout algorithms."""

    SPRING = "spring"
    CIRCULAR = "circular"
    RANDOM = "random"
    KAMADA_KAWAI = "kamada_kawai"
    SPECTRAL = "spectral"
    SHELL = "shell"
    PLANAR = "planar"


class VisualizationBackend(str, Enum):
    """Supported visualization backends."""

    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    BOKEH = "bokeh"
    PYVIS = "pyvis"


@dataclass
class ColorScheme:
    """Color scheme configuration for graph visualization."""

    name: str
    node_colors: dict[str, str] = field(default_factory=dict)
    edge_color: str = "#888888"
    background_color: str = "#ffffff"
    text_color: str = "#000000"

    @classmethod
    def default_entity_colors(cls) -> "ColorScheme":
        """Create default color scheme for entity types."""
        return cls(
            name="Entity Types",
            node_colors={
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


@dataclass
class LayoutConfig:
    """Configuration for graph layout algorithms."""

    algorithm: LayoutAlgorithm = LayoutAlgorithm.SPRING
    iterations: int | None = None
    k: float | None = None  # Optimal distance between nodes
    seed: int | None = None  # Random seed for reproducibility
    additional_params: dict[str, Any] = field(default_factory=dict)

    def to_networkx_params(self) -> dict[str, Any]:
        """Convert to parameters suitable for NetworkX layout functions."""
        params = dict(self.additional_params)

        if self.iterations is not None:
            params["iterations"] = self.iterations
        if self.k is not None:
            params["k"] = self.k
        if self.seed is not None:
            params["seed"] = self.seed

        return params


@dataclass
class NodeStyling:
    """Node styling configuration."""

    size_attribute: str | None = None
    color_attribute: str | None = "entity_type"
    min_size: float = 10.0
    max_size: float = 50.0
    default_size: float = 20.0
    default_color: str = "#1f78b4"
    alpha: float = 0.8
    border_width: float = 1.0
    border_color: str = "#ffffff"


@dataclass
class EdgeStyling:
    """Edge styling configuration."""

    width_attribute: str | None = None
    color_attribute: str | None = None
    min_width: float = 0.5
    max_width: float = 5.0
    default_width: float = 1.0
    default_color: str = "#888888"
    alpha: float = 0.6
    style: str = "solid"  # solid, dashed, dotted


@dataclass
class LabelConfig:
    """Label configuration for nodes and edges."""

    show_node_labels: bool = True
    show_edge_labels: bool = False
    node_label_attribute: str | None = None
    edge_label_attribute: str | None = None
    font_size: int = 10
    font_color: str = "#000000"
    font_family: str = "Arial"
    max_label_length: int = 20


@dataclass
class InteractivityConfig:
    """Configuration for interactive features."""

    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    enable_selection: bool = True
    hover_info: list[str] = field(default_factory=lambda: ["id", "entity_type"])
    click_behavior: str = "select"  # select, highlight, info


@dataclass
class VisualizationConfig:
    """Complete configuration for graph visualization."""

    backend: VisualizationBackend = VisualizationBackend.PLOTLY
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    color_scheme: ColorScheme = field(default_factory=ColorScheme.default_entity_colors)
    node_styling: NodeStyling = field(default_factory=NodeStyling)
    edge_styling: EdgeStyling = field(default_factory=EdgeStyling)
    labels: LabelConfig = field(default_factory=LabelConfig)
    interactivity: InteractivityConfig = field(default_factory=InteractivityConfig)

    # Figure dimensions
    width: int = 1200
    height: int = 800
    title: str = "Knowledge Graph Visualization"

    # Performance settings
    max_nodes_for_labels: int = 100
    max_nodes_for_interactive: int = 500

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "backend": self.backend.value,
            "layout": {
                "algorithm": self.layout.algorithm.value,
                "iterations": self.layout.iterations,
                "k": self.layout.k,
                "seed": self.layout.seed,
                "additional_params": self.layout.additional_params,
            },
            "color_scheme": {
                "name": self.color_scheme.name,
                "node_colors": self.color_scheme.node_colors,
                "edge_color": self.color_scheme.edge_color,
                "background_color": self.color_scheme.background_color,
                "text_color": self.color_scheme.text_color,
            },
            "node_styling": {
                "size_attribute": self.node_styling.size_attribute,
                "color_attribute": self.node_styling.color_attribute,
                "min_size": self.node_styling.min_size,
                "max_size": self.node_styling.max_size,
                "default_size": self.node_styling.default_size,
                "default_color": self.node_styling.default_color,
                "alpha": self.node_styling.alpha,
                "border_width": self.node_styling.border_width,
                "border_color": self.node_styling.border_color,
            },
            "edge_styling": {
                "width_attribute": self.edge_styling.width_attribute,
                "color_attribute": self.edge_styling.color_attribute,
                "min_width": self.edge_styling.min_width,
                "max_width": self.edge_styling.max_width,
                "default_width": self.edge_styling.default_width,
                "default_color": self.edge_styling.default_color,
                "alpha": self.edge_styling.alpha,
                "style": self.edge_styling.style,
            },
            "labels": {
                "show_node_labels": self.labels.show_node_labels,
                "show_edge_labels": self.labels.show_edge_labels,
                "node_label_attribute": self.labels.node_label_attribute,
                "edge_label_attribute": self.labels.edge_label_attribute,
                "font_size": self.labels.font_size,
                "font_color": self.labels.font_color,
                "font_family": self.labels.font_family,
                "max_label_length": self.labels.max_label_length,
            },
            "interactivity": {
                "enable_zoom": self.interactivity.enable_zoom,
                "enable_pan": self.interactivity.enable_pan,
                "enable_hover": self.interactivity.enable_hover,
                "enable_selection": self.interactivity.enable_selection,
                "hover_info": self.interactivity.hover_info,
                "click_behavior": self.interactivity.click_behavior,
            },
            "width": self.width,
            "height": self.height,
            "title": self.title,
            "max_nodes_for_labels": self.max_nodes_for_labels,
            "max_nodes_for_interactive": self.max_nodes_for_interactive,
        }
