"""
Graph visualization module for NetworkX graphs.
"""

import logging

import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize NetworkX graphs with various layout algorithms and styling options."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize the visualizer.

        Args:
            graph: NetworkX graph to visualize
        """
        self.graph = graph
        self.node_colors = {}
        self.edge_colors = {}
        self.node_sizes = {}
        self.positions = {}

    def set_node_colors_by_attribute(
        self, attribute: str, colormap: str = "tab10"
    ) -> dict[str, str]:
        """
        Set node colors based on a node attribute.

        Args:
            attribute: Node attribute to use for coloring
            colormap: Matplotlib colormap name

        Returns:
            Dictionary mapping attribute values to colors
        """
        # Get unique values for the attribute
        unique_values = set()
        for node, data in self.graph.nodes(data=True):
            if attribute in data and data[attribute] is not None:
                unique_values.add(data[attribute])

        unique_values = sorted(list(unique_values))

        # Generate colors
        if len(unique_values) <= 10:
            colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(unique_values)))
        else:
            colors = cc.glasbey[: len(unique_values)]

        color_map = {
            val: f"#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}"
            if len(c) >= 3
            else "#999999"
            for val, c in zip(unique_values, colors, strict=False)
        }

        # Assign colors to nodes
        for node, data in self.graph.nodes(data=True):
            attr_value = data.get(attribute, "Unknown")
            self.node_colors[node] = color_map.get(attr_value, "#999999")

        return color_map

    def set_node_sizes_by_attribute(
        self, attribute: str, min_size: int = 20, max_size: int = 200
    ):
        """
        Set node sizes based on a node attribute.

        Args:
            attribute: Node attribute to use for sizing
            min_size: Minimum node size
            max_size: Maximum node size
        """
        values = []
        for node, data in self.graph.nodes(data=True):
            if attribute in data and data[attribute] is not None:
                try:
                    values.append(float(data[attribute]))
                except (ValueError, TypeError):
                    values.append(0)
            else:
                values.append(0)

        if values and max(values) > min(values):
            min_val, max_val = min(values), max(values)
            for i, (node, _) in enumerate(self.graph.nodes(data=True)):
                normalized = (values[i] - min_val) / (max_val - min_val)
                self.node_sizes[node] = min_size + normalized * (max_size - min_size)
        else:
            for node in self.graph.nodes():
                self.node_sizes[node] = (min_size + max_size) / 2

    def compute_layout(
        self, layout: str = "spring", **kwargs
    ) -> dict[str, tuple[float, float]]:
        """
        Compute node positions using specified layout algorithm.

        Args:
            layout: Layout algorithm name
            **kwargs: Additional arguments for layout algorithm

        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        logger.info(
            f"Computing {layout} layout for {self.graph.number_of_nodes()} nodes"
        )

        layout_functions = {
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
            "random": nx.random_layout,
            "shell": nx.shell_layout,
            "spectral": nx.spectral_layout,
            "kamada_kawai": nx.kamada_kawai_layout,
            "planar": nx.planar_layout,
        }

        if layout not in layout_functions:
            logger.warning(f"Unknown layout '{layout}', using 'spring' instead")
            layout = "spring"

        try:
            self.positions = layout_functions[layout](self.graph, **kwargs)
        except Exception as e:
            logger.warning(
                f"Layout '{layout}' failed: {e}, falling back to spring layout"
            )
            self.positions = nx.spring_layout(self.graph)

        return self.positions

    def plot_matplotlib(
        self,
        figsize: tuple[int, int] = (12, 8),
        layout: str = "spring",
        show_labels: bool = True,
        node_label_attr: str | None = None,
        title: str = "Knowledge Graph",
        **kwargs,
    ) -> plt.Figure:
        """
        Create a matplotlib visualization of the graph.

        Args:
            figsize: Figure size
            layout: Layout algorithm
            show_labels: Whether to show node labels
            node_label_attr: Node attribute to use for labels
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            Matplotlib figure
        """
        if not self.positions:
            self.compute_layout(layout)

        fig, ax = plt.subplots(figsize=figsize)

        # Draw edges
        edge_colors = [
            self.edge_colors.get(edge, "#cccccc") for edge in self.graph.edges()
        ]
        nx.draw_networkx_edges(
            self.graph, self.positions, edge_color=edge_colors, alpha=0.6, ax=ax
        )

        # Draw nodes
        node_colors = [
            self.node_colors.get(node, "#1f78b4") for node in self.graph.nodes()
        ]
        node_sizes = [self.node_sizes.get(node, 50) for node in self.graph.nodes()]

        nx.draw_networkx_nodes(
            self.graph,
            self.positions,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax,
        )

        # Draw labels
        if show_labels:
            if node_label_attr:
                labels = {
                    node: data.get(node_label_attr, node)
                    for node, data in self.graph.nodes(data=True)
                }
            else:
                labels = {node: node for node in self.graph.nodes()}

            nx.draw_networkx_labels(
                self.graph, self.positions, labels, font_size=8, ax=ax
            )

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()

        return fig

    def plot_plotly(
        self,
        layout: str = "spring",
        node_label_attr: str | None = None,
        edge_label_attr: str | None = None,
        title: str = "Interactive Knowledge Graph",
        width: int = 1200,
        height: int = 800,
    ) -> go.Figure:
        """
        Create an interactive plotly visualization of the graph.

        Args:
            layout: Layout algorithm
            node_label_attr: Node attribute to use for labels
            edge_label_attr: Edge attribute to use for edge labels
            title: Plot title
            width: Figure width
            height: Figure height

        Returns:
            Plotly figure
        """
        if not self.positions:
            self.compute_layout(layout)

        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []

        for edge in self.graph.edges(data=True):
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Edge info for hover
            edge_text = f"{edge[0]} → {edge[1]}"
            if edge_label_attr and edge_label_attr in edge[2]:
                edge_text += f"<br>{edge_label_attr}: {edge[2][edge_label_attr]}"
            edge_info.append(edge_text)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_colors = []
        node_sizes = []

        for node in self.graph.nodes(data=True):
            x, y = self.positions[node[0]]
            node_x.append(x)
            node_y.append(y)

            # Node label
            if node_label_attr and node_label_attr in node[1]:
                label = str(node[1][node_label_attr])[:50]  # Truncate long labels
            else:
                label = str(node[0])[:50]
            node_text.append(label)

            # Node hover info
            info_parts = [f"ID: {node[0]}"]
            for key, value in node[1].items():
                if key != node_label_attr and value is not None:
                    info_parts.append(f"{key}: {str(value)[:100]}")
            node_info.append("<br>".join(info_parts))

            # Node styling
            node_colors.append(self.node_colors.get(node[0], "#1f78b4"))
            node_sizes.append(max(10, self.node_sizes.get(node[0], 20)))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10),
            marker=dict(
                size=node_sizes, color=node_colors, line=dict(width=1, color="white")
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=title, font=dict(size=16)),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height,
            ),
        )

        return fig

    def create_legend(self, attribute: str, color_map: dict[str, str]) -> go.Figure:
        """
        Create a legend for node colors.

        Args:
            attribute: Attribute name
            color_map: Color mapping

        Returns:
            Plotly figure with legend
        """
        fig = go.Figure()

        for value, color in color_map.items():
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode="markers",
                    marker=dict(size=20, color=color),
                    name=str(value),
                    showlegend=True,
                )
            )

        fig.update_layout(
            title=f"Legend: {attribute}",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=min(600, len(color_map) * 30 + 100),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return fig
