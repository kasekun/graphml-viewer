"""
Command-line interface for the GraphML viewer.
"""

import argparse
import logging
import sys
from pathlib import Path

from .analysis import GraphAnalyzer
from .graph_loader import GraphMLLoader
from .visualizer import GraphVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="GraphML Viewer - Visualize and explore LightRAG knowledge graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive web interface
  graphml-viewer --interactive

  # Generate static visualization
  graphml-viewer graph.graphml --output viz.png

  # Analyze graph and show statistics
  graphml-viewer graph.graphml --analyze

  # Create visualization with custom layout
  graphml-viewer graph.graphml --layout spring --output spring_layout.html
        """,
    )

    parser.add_argument(
        "graphml_file", nargs="?", help="Path to the GraphML file to visualize"
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch interactive Streamlit web interface",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file path for visualization (supports .png, .pdf, .html)",
    )

    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "circular", "random", "kamada_kawai", "spectral"],
        help="Layout algorithm for node positioning",
    )

    parser.add_argument(
        "-a",
        "--analyze",
        action="store_true",
        help="Perform graph analysis and print statistics",
    )

    parser.add_argument(
        "--color-by",
        type=str,
        default="entity_type",
        help="Node attribute to use for coloring",
    )

    parser.add_argument("--size-by", type=str, help="Node attribute to use for sizing")

    parser.add_argument(
        "--filter-types",
        type=str,
        nargs="+",
        help="Filter to include only specific entity types",
    )

    parser.add_argument(
        "--width", type=int, default=1200, help="Width of the output visualization"
    )

    parser.add_argument(
        "--height", type=int, default=800, help="Height of the output visualization"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    return parser


def load_and_prepare_graph(file_path: str, filter_types: list | None = None) -> tuple:
    """Load and prepare the graph for visualization."""
    logger.info(f"Loading GraphML file: {file_path}")

    # Load the graph
    loader = GraphMLLoader(file_path)
    graph = loader.load()

    # Filter by entity types if specified
    if filter_types:
        logger.info(f"Filtering to entity types: {filter_types}")
        graph = loader.filter_by_entity_type(filter_types)
        logger.info(
            f"Filtered graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

    # Create visualizer and analyzer
    visualizer = GraphVisualizer(graph)
    analyzer = GraphAnalyzer(graph)

    return loader, visualizer, analyzer


def run_analysis(loader: GraphMLLoader, analyzer: GraphAnalyzer):
    """Run and display graph analysis."""
    logger.info("Performing graph analysis...")

    # Basic statistics
    stats = loader.get_graph_statistics()

    print("\n" + "=" * 50)
    print("GRAPH STATISTICS")
    print("=" * 50)
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Density: {stats['density']:.4f}")
    print(f"Connected: {stats['is_connected']}")
    print(f"Connected components: {stats['num_connected_components']}")

    if stats.get("avg_degree"):
        print(f"Average degree: {stats['avg_degree']:.2f}")
        print(f"Max degree: {stats['max_degree']}")
        print(f"Min degree: {stats['min_degree']}")

    # Entity type distribution
    if "entity_type_counts" in stats:
        print("\nEntity Type Distribution:")
        for entity_type, count in stats["entity_type_counts"].items():
            print(f"  {entity_type}: {count}")

    # Top nodes by centrality
    print("\nTop 10 nodes by PageRank centrality:")
    try:
        ranking = analyzer.get_node_importance_ranking("pagerank", 10)
        for i, (node, score) in enumerate(ranking, 1):
            print(f"  {i:2d}. {node} ({score:.4f})")
    except Exception as e:
        print(f"  Error computing centrality: {e}")

    # Community detection
    print("\nCommunity Detection:")
    try:
        communities = analyzer.find_communities("greedy")
        community_sizes = {}
        for node, community_id in communities.items():
            community_sizes[community_id] = community_sizes.get(community_id, 0) + 1

        print(f"  Found {len(community_sizes)} communities")
        for i, (community_id, size) in enumerate(
            sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
        ):
            print(f"    Community {community_id}: {size} nodes")

    except Exception as e:
        print(f"  Error in community detection: {e}")

    print("=" * 50)


def create_visualization(
    visualizer: GraphVisualizer,
    output_path: str,
    layout: str = "spring",
    color_by: str = "entity_type",
    size_by: str | None = None,
    width: int = 1200,
    height: int = 800,
):
    """Create and save visualization."""
    logger.info(f"Creating visualization with {layout} layout...")

    # Set up styling
    if color_by:
        try:
            color_map = visualizer.set_node_colors_by_attribute(color_by)
            logger.info(f"Colored nodes by {color_by} ({len(color_map)} unique values)")
        except Exception as e:
            logger.warning(f"Could not color by {color_by}: {e}")

    if size_by:
        try:
            visualizer.set_node_sizes_by_attribute(size_by)
            logger.info(f"Sized nodes by {size_by}")
        except Exception as e:
            logger.warning(f"Could not size by {size_by}: {e}")

    # Determine output format
    output_path_obj = Path(output_path)
    output_format = output_path_obj.suffix.lower()

    if output_format in [".html"]:
        # Create interactive Plotly visualization
        fig = visualizer.plot_plotly(
            layout=layout,
            title="Knowledge Graph Visualization",
            width=width,
            height=height,
        )
        fig.write_html(str(output_path_obj))
        logger.info(f"Saved interactive visualization to {output_path_obj}")

    elif output_format in [".png", ".pdf", ".svg"]:
        # Create static matplotlib visualization
        fig = visualizer.plot_matplotlib(
            layout=layout,
            figsize=(width // 100, height // 100),
            title="Knowledge Graph Visualization",
        )
        fig.savefig(str(output_path_obj), dpi=150, bbox_inches="tight")
        logger.info(f"Saved static visualization to {output_path_obj}")

    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle interactive mode
    if args.interactive:
        logger.info("Launching interactive Streamlit interface...")
        try:
            import subprocess

            # Get the path to the standalone streamlit app
            current_dir = Path(__file__).parent.parent.parent
            streamlit_app_path = current_dir / "streamlit_app.py"

            if not streamlit_app_path.exists():
                logger.error(f"Streamlit app not found at: {streamlit_app_path}")
                sys.exit(1)

            # Launch streamlit
            cmd = ["streamlit", "run", str(streamlit_app_path)]
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

        except ImportError:
            logger.error(
                "Streamlit not available. Please install with: uv add streamlit"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error launching interactive interface: {e}")
            sys.exit(1)

    # Handle file-based operations
    elif args.graphml_file:
        file_path = Path(args.graphml_file)

        if not file_path.exists():
            logger.error(f"GraphML file not found: {file_path}")
            sys.exit(1)

        try:
            # Load and prepare graph
            loader, visualizer, analyzer = load_and_prepare_graph(
                str(file_path), args.filter_types
            )

            # Run analysis if requested
            if args.analyze:
                run_analysis(loader, analyzer)

            # Create visualization if output specified
            if args.output:
                create_visualization(
                    visualizer,
                    args.output,
                    args.layout,
                    args.color_by,
                    args.size_by,
                    args.width,
                    args.height,
                )

            # If neither analysis nor output specified, just show basic info
            if not args.analyze and not args.output:
                stats = loader.get_graph_statistics()
                print(
                    f"Loaded graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges"
                )
                print(
                    "Use --analyze for detailed analysis or --output to create visualization"
                )

        except Exception as e:
            logger.error(f"Error processing GraphML file: {e}")
            sys.exit(1)

    else:
        # No file specified and not interactive mode
        parser.print_help()
        print(
            "\nHint: Use --interactive to launch the web interface, or specify a GraphML file to analyze"
        )


if __name__ == "__main__":
    main()
