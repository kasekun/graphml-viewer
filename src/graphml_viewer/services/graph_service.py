"""
Graph service for handling graph operations and data management.
"""

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from ..models.graph_models import GraphStatistics


class GraphService:
    """Service for graph operations and data management."""

    def __init__(self) -> None:
        """Initialize the graph service."""
        self.current_graph: nx.Graph | None = None
        self.graph_metadata: dict[str, Any] = {}

    def load_graphml_file(self, file_path: str | Path) -> nx.Graph:
        """
        Load a GraphML file into memory.

        Args:
            file_path: Path to the GraphML file

        Returns:
            Loaded NetworkX graph

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"GraphML file not found: {file_path}")

        try:
            self.current_graph = nx.read_graphml(str(file_path))
            if self.current_graph is None:
                raise ValueError("Failed to load graph from file")

            # Store metadata
            self.graph_metadata = {
                "source_file": str(file_path),
                "loaded_at": pd.Timestamp.now().isoformat(),
                "num_nodes": self.current_graph.number_of_nodes(),
                "num_edges": self.current_graph.number_of_edges(),
            }

            return self.current_graph

        except Exception as e:
            raise ValueError(f"Failed to parse GraphML file: {e}") from e

    def get_current_graph(self) -> nx.Graph | None:
        """Get the currently loaded graph."""
        return self.current_graph

    def validate_graph_loaded(self) -> nx.Graph:
        """Validate that a graph is loaded and return it."""
        if self.current_graph is None:
            raise ValueError("No graph loaded. Load a graph first.")
        return self.current_graph

    def get_graph_statistics(self) -> GraphStatistics:
        """
        Get comprehensive statistics about the current graph.

        Returns:
            GraphStatistics object with all relevant metrics
        """
        graph = self.validate_graph_loaded()
        stats = GraphStatistics(
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            density=nx.density(graph),
            is_directed=nx.is_directed(graph),
            is_connected=nx.is_connected(graph),
            num_connected_components=nx.number_connected_components(graph),
        )

        # Add degree statistics if graph has nodes
        if graph.number_of_nodes() > 0:
            degrees = [degree for node, degree in graph.degree()]
            if degrees:
                stats.avg_degree = sum(degrees) / len(degrees)
                stats.max_degree = max(degrees)
                stats.min_degree = min(degrees)

        # Add entity type distribution
        entity_types = self.get_entity_type_distribution()
        if entity_types:
            stats.entity_type_counts = entity_types

        return stats

    def get_node_data_frame(self) -> pd.DataFrame:
        """
        Extract node data as a pandas DataFrame.

        Returns:
            DataFrame with node attributes
        """
        self.validate_graph_loaded()

        nodes_data = []
        for node_id, data in self.current_graph.nodes(data=True):
            row = {"id": node_id}
            row.update(data)
            nodes_data.append(row)

        return pd.DataFrame(nodes_data)

    def get_edge_data_frame(self) -> pd.DataFrame:
        """
        Extract edge data as a pandas DataFrame.

        Returns:
            DataFrame with edge attributes
        """
        self.validate_graph_loaded()

        edges_data = []
        for source, target, data in self.current_graph.edges(data=True):
            row = {"source": source, "target": target}
            row.update(data)
            edges_data.append(row)

        return pd.DataFrame(edges_data)

    def get_entity_types(self) -> list[str]:
        """
        Get unique entity types from the graph.

        Returns:
            Sorted list of entity types
        """
        self.validate_graph_loaded()

        entity_types = set()
        for node_id, data in self.current_graph.nodes(data=True):
            entity_type = data.get("entity_type")
            if entity_type is not None:
                entity_types.add(entity_type)

        return sorted(list(entity_types))

    def get_entity_type_distribution(self) -> dict[str, int]:
        """
        Get the distribution of entity types in the graph.

        Returns:
            Dictionary mapping entity types to counts
        """
        node_df = self.get_node_data_frame()
        if "entity_type" in node_df.columns:
            return node_df["entity_type"].value_counts().to_dict()
        return {}

    def filter_by_entity_types(self, entity_types: list[str]) -> nx.Graph:
        """
        Create a subgraph containing only nodes of specified entity types.

        Args:
            entity_types: List of entity types to include

        Returns:
            Filtered NetworkX graph
        """
        self.validate_graph_loaded()

        nodes_to_include = []
        for node_id, data in self.current_graph.nodes(data=True):
            if data.get("entity_type") in entity_types:
                nodes_to_include.append(node_id)

        return self.current_graph.subgraph(nodes_to_include).copy()

    def filter_by_node_attributes(self, **filters) -> nx.Graph:
        """
        Filter nodes by multiple attributes.

        Args:
            **filters: Attribute name and value pairs to filter by

        Returns:
            Filtered NetworkX graph
        """
        self.validate_graph_loaded()

        nodes_to_include = []
        for node_id, data in self.current_graph.nodes(data=True):
            include_node = True
            for attr_name, attr_value in filters.items():
                if data.get(attr_name) != attr_value:
                    include_node = False
                    break
            if include_node:
                nodes_to_include.append(node_id)

        return self.current_graph.subgraph(nodes_to_include).copy()

    def get_subgraph_by_distance(
        self, center_node: str, max_distance: int = 2
    ) -> nx.Graph:
        """
        Extract a subgraph containing nodes within a certain distance from center.

        Args:
            center_node: Central node ID
            max_distance: Maximum distance from center node

        Returns:
            Subgraph containing nodes within max_distance
        """
        self.validate_graph_loaded()

        if center_node not in self.current_graph:
            raise ValueError(f"Node {center_node} not found in graph")

        # Find all nodes within max_distance using BFS
        visited = set()
        current_level = {center_node}
        visited.add(center_node)

        for distance in range(max_distance):
            next_level = set()
            for node in current_level:
                neighbors = set(self.current_graph.neighbors(node))
                next_level.update(neighbors - visited)

            visited.update(next_level)
            current_level = next_level

            if not current_level:  # No more nodes to explore
                break

        return self.current_graph.subgraph(visited).copy()

    def sample_nodes(self, n_nodes: int, method: str = "random") -> nx.Graph:
        """
        Sample a subset of nodes from the graph.

        Args:
            n_nodes: Number of nodes to sample
            method: Sampling method ("random", "degree", "pagerank")

        Returns:
            Subgraph with sampled nodes
        """
        self.validate_graph_loaded()

        total_nodes = self.current_graph.number_of_nodes()
        if n_nodes >= total_nodes:
            return self.current_graph.copy()

        if method == "random":
            import random

            nodes = random.sample(list(self.current_graph.nodes()), n_nodes)

        elif method == "degree":
            # Sample nodes with highest degree
            degrees = dict(self.current_graph.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            nodes = [node for node, degree in sorted_nodes[:n_nodes]]

        elif method == "pagerank":
            # Sample nodes with highest PageRank
            pagerank = nx.pagerank(self.current_graph)
            sorted_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            nodes = [node for node, score in sorted_nodes[:n_nodes]]

        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return self.current_graph.subgraph(nodes).copy()

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata about the current graph."""
        return self.graph_metadata.copy()
