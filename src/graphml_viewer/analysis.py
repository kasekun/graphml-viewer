"""
Graph analysis module for computing graph metrics and insights.
"""

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """Analyze NetworkX graphs to compute various metrics and insights."""

    def __init__(self, graph: nx.Graph):
        """
        Initialize the analyzer.

        Args:
            graph: NetworkX graph to analyze
        """
        self.graph = graph
        self._centrality_cache = {}

    def compute_centrality_measures(self) -> dict[str, dict[str, float]]:
        """
        Compute various centrality measures for all nodes.

        Returns:
            Dictionary of centrality measures
        """
        if "centrality" not in self._centrality_cache:
            logger.info("Computing centrality measures...")

            centrality_measures = {}

            # Degree centrality
            centrality_measures["degree"] = nx.degree_centrality(self.graph)

            # Betweenness centrality (can be slow for large graphs)
            if self.graph.number_of_nodes() < 1000:
                centrality_measures["betweenness"] = nx.betweenness_centrality(
                    self.graph
                )
            else:
                # Sample for large graphs
                k = min(100, self.graph.number_of_nodes())
                centrality_measures["betweenness"] = nx.betweenness_centrality(
                    self.graph, k=k, normalized=True
                )

            # Closeness centrality
            if nx.is_connected(self.graph):
                centrality_measures["closeness"] = nx.closeness_centrality(self.graph)
            else:
                # For disconnected graphs, compute for largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                closeness = nx.closeness_centrality(subgraph)
                # Fill in zeros for nodes not in largest component
                centrality_measures["closeness"] = {
                    node: closeness.get(node, 0.0) for node in self.graph.nodes()
                }

            # Eigenvector centrality (if possible)
            try:
                centrality_measures["eigenvector"] = nx.eigenvector_centrality(
                    self.graph, max_iter=1000
                )
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge")
                centrality_measures["eigenvector"] = dict.fromkeys(
                    self.graph.nodes(), 0.0
                )

            # PageRank
            centrality_measures["pagerank"] = nx.pagerank(self.graph)

            self._centrality_cache["centrality"] = centrality_measures

        return self._centrality_cache["centrality"]

    def find_communities(self, method: str = "louvain") -> dict[str, int]:
        """
        Detect communities in the graph.

        Args:
            method: Community detection method ("louvain", "greedy", "label_propagation")

        Returns:
            Dictionary mapping node IDs to community IDs
        """
        logger.info(f"Detecting communities using {method} method...")

        if method == "louvain":
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(self.graph)
                return partition
            except ImportError:
                logger.warning(
                    "python-louvain not available, falling back to greedy method"
                )
                method = "greedy"

        if method == "greedy":
            communities = nx.community.greedy_modularity_communities(self.graph)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            return partition

        elif method == "label_propagation":
            communities = nx.community.label_propagation_communities(self.graph)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            return partition

        else:
            raise ValueError(f"Unknown community detection method: {method}")

    def get_node_importance_ranking(
        self, centrality_measure: str = "pagerank", top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Rank nodes by importance based on centrality measure.

        Args:
            centrality_measure: Which centrality measure to use
            top_k: Number of top nodes to return (None for all)

        Returns:
            List of (node_id, centrality_score) tuples, sorted by score
        """
        centrality_measures = self.compute_centrality_measures()

        if centrality_measure not in centrality_measures:
            raise ValueError(f"Unknown centrality measure: {centrality_measure}")

        centrality_scores = centrality_measures[centrality_measure]
        ranking = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

        if top_k is not None:
            ranking = ranking[:top_k]

        return ranking

    def analyze_entity_type_connectivity(self) -> dict[str, Any]:
        """
        Analyze connectivity patterns between different entity types.

        Returns:
            Dictionary with connectivity analysis results
        """
        # Get entity types for each node
        entity_types = {}
        for node, data in self.graph.nodes(data=True):
            entity_types[node] = data.get("entity_type", "unknown")

        # Count connections between entity types
        type_connections = {}
        for source, target in self.graph.edges():
            source_type = entity_types[source]
            target_type = entity_types[target]

            # Create sorted tuple to avoid double counting
            connection_type = tuple(sorted([source_type, target_type]))

            if connection_type not in type_connections:
                type_connections[connection_type] = 0
            type_connections[connection_type] += 1

        # Calculate type-specific metrics
        type_metrics = {}
        unique_types = set(entity_types.values())

        for entity_type in unique_types:
            nodes_of_type = [
                node for node, t in entity_types.items() if t == entity_type
            ]
            subgraph = self.graph.subgraph(nodes_of_type)

            type_metrics[entity_type] = {
                "count": len(nodes_of_type),
                "internal_edges": subgraph.number_of_edges(),
                "avg_degree": np.mean([d for n, d in subgraph.degree()])
                if nodes_of_type
                else 0,
            }

        return {
            "type_connections": type_connections,
            "type_metrics": type_metrics,
            "entity_type_distribution": {
                t: len([n for n, et in entity_types.items() if et == t])
                for t in unique_types
            },
        }

    def get_subgraph_by_distance(
        self, center_node: str, max_distance: int = 2
    ) -> nx.Graph:
        """
        Extract a subgraph containing nodes within a certain distance from a center node.

        Args:
            center_node: Central node ID
            max_distance: Maximum distance from center node

        Returns:
            Subgraph containing nodes within max_distance
        """
        if center_node not in self.graph:
            raise ValueError(f"Node {center_node} not found in graph")

        # Find all nodes within max_distance
        nodes_in_subgraph = set([center_node])

        for distance in range(1, max_distance + 1):
            new_nodes = set()
            for node in nodes_in_subgraph:
                neighbors = set(self.graph.neighbors(node))
                new_nodes.update(neighbors)
            nodes_in_subgraph.update(new_nodes)

        return self.graph.subgraph(nodes_in_subgraph).copy()

    def find_shortest_paths(
        self, source: str, target: str | None = None, max_paths: int = 5
    ) -> dict[str, Any]:
        """
        Find shortest paths between nodes.

        Args:
            source: Source node ID
            target: Target node ID (if None, finds paths to all nodes)
            max_paths: Maximum number of paths to return

        Returns:
            Dictionary with path information
        """
        if source not in self.graph:
            raise ValueError(f"Source node {source} not found in graph")

        if target is not None:
            if target not in self.graph:
                raise ValueError(f"Target node {target} not found in graph")

            try:
                path = nx.shortest_path(self.graph, source, target)
                length = len(path) - 1
                return {
                    "source": source,
                    "target": target,
                    "path": path,
                    "length": length,
                }
            except nx.NetworkXNoPath:
                return {
                    "source": source,
                    "target": target,
                    "path": None,
                    "length": float("inf"),
                }
        else:
            # Find shortest paths to all reachable nodes
            try:
                paths = nx.single_source_shortest_path(self.graph, source)
                path_info = []

                for target_node, path in paths.items():
                    if target_node != source:  # Skip self
                        path_info.append(
                            {
                                "target": target_node,
                                "path": path,
                                "length": len(path) - 1,
                            }
                        )

                # Sort by path length and limit
                path_info.sort(key=lambda x: x["length"])
                path_info = path_info[:max_paths]

                return {"source": source, "paths": path_info}
            except Exception as e:
                logger.error(f"Error computing shortest paths: {e}")
                return {"source": source, "paths": []}
