"""
Data models for graph structures and analysis results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NodeData:
    """Represents a single node in the graph."""

    id: str
    entity_type: str | None = None
    description: str | None = None
    source_id: str | None = None
    file_path: str | None = None
    created_at: str | None = None
    additional_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "entity_type": self.entity_type,
            "description": self.description,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "created_at": self.created_at,
        }
        result.update(self.additional_attributes)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class EdgeData:
    """Represents a single edge in the graph."""

    source: str
    target: str
    weight: float | None = None
    description: str | None = None
    keywords: str | None = None
    source_id: str | None = None
    file_path: str | None = None
    created_at: str | None = None
    additional_attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "description": self.description,
            "keywords": self.keywords,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "created_at": self.created_at,
        }
        result.update(self.additional_attributes)
        return {k: v for k, v in result.items() if v is not None}


@dataclass
class GraphStatistics:
    """Statistics about a graph."""

    num_nodes: int
    num_edges: int
    density: float
    is_directed: bool
    is_connected: bool
    num_connected_components: int
    avg_degree: float | None = None
    max_degree: int | None = None
    min_degree: int | None = None
    entity_type_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.density,
            "is_directed": self.is_directed,
            "is_connected": self.is_connected,
            "num_connected_components": self.num_connected_components,
            "avg_degree": self.avg_degree,
            "max_degree": self.max_degree,
            "min_degree": self.min_degree,
            "entity_type_counts": self.entity_type_counts,
        }


@dataclass
class CentralityMeasures:
    """Centrality measures for graph nodes."""

    degree: dict[str, float] = field(default_factory=dict)
    betweenness: dict[str, float] = field(default_factory=dict)
    closeness: dict[str, float] = field(default_factory=dict)
    eigenvector: dict[str, float] = field(default_factory=dict)
    pagerank: dict[str, float] = field(default_factory=dict)


@dataclass
class CommunityInfo:
    """Community detection results."""

    algorithm: str
    num_communities: int
    modularity: float | None = None
    node_assignments: dict[str, int] = field(default_factory=dict)
    community_sizes: dict[int, int] = field(default_factory=dict)


@dataclass
class ConnectivityAnalysis:
    """Entity type connectivity analysis results."""

    type_connections: dict[tuple[str, str], int] = field(default_factory=dict)
    type_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    entity_type_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result for a graph."""

    statistics: GraphStatistics
    centrality_measures: CentralityMeasures | None = None
    community_info: CommunityInfo | None = None
    connectivity_analysis: ConnectivityAnalysis | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {"statistics": self.statistics.to_dict()}

        if self.centrality_measures:
            result["centrality_measures"] = {
                "degree": self.centrality_measures.degree,
                "betweenness": self.centrality_measures.betweenness,
                "closeness": self.centrality_measures.closeness,
                "eigenvector": self.centrality_measures.eigenvector,
                "pagerank": self.centrality_measures.pagerank,
            }

        if self.community_info:
            result["community_info"] = {
                "algorithm": self.community_info.algorithm,
                "num_communities": self.community_info.num_communities,
                "modularity": self.community_info.modularity,
                "node_assignments": self.community_info.node_assignments,
                "community_sizes": self.community_info.community_sizes,
            }

        if self.connectivity_analysis:
            result["connectivity_analysis"] = {
                "type_connections": {
                    f"{k[0]}-{k[1]}": v
                    for k, v in self.connectivity_analysis.type_connections.items()
                },
                "type_metrics": self.connectivity_analysis.type_metrics,
                "entity_type_distribution": self.connectivity_analysis.entity_type_distribution,
            }

        return result
