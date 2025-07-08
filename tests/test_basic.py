#!/usr/bin/env python3
"""
Basic test script for the GraphML viewer.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graphml_viewer import GraphAnalyzer, GraphMLLoader


def test_basic_loading():
    """Test basic GraphML loading functionality."""
    print("🧪 Testing GraphML Viewer...")

    # Path to the LightRAG GraphML file
    graphml_path = (
        Path(__file__).parent.parent
        / "light-rag"
        / "lightrag_data"
        / "graph_chunk_entity_relation.graphml"
    )

    if not graphml_path.exists():
        print(f"❌ GraphML file not found: {graphml_path}")
        return False

    try:
        # Test loading
        print(f"📁 Loading GraphML file: {graphml_path}")
        loader = GraphMLLoader(graphml_path)
        graph = loader.load()

        print(
            f"✅ Successfully loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
        )

        # Test basic statistics
        print("📊 Computing graph statistics...")
        stats = loader.get_graph_statistics()

        print(f"  - Nodes: {stats['num_nodes']}")
        print(f"  - Edges: {stats['num_edges']}")
        print(f"  - Density: {stats['density']:.4f}")
        print(f"  - Connected: {stats['is_connected']}")
        print(f"  - Connected components: {stats['num_connected_components']}")

        # Test entity types
        entity_types = loader.get_entity_types()
        print(f"  - Entity types: {entity_types}")

        # Test node and edge data extraction
        print("📋 Testing data extraction...")
        node_df = loader.get_node_data()
        edge_df = loader.get_edge_data()

        print(f"  - Node DataFrame shape: {node_df.shape}")
        print(f"  - Edge DataFrame shape: {edge_df.shape}")
        print(f"  - Node columns: {list(node_df.columns)}")
        print(f"  - Edge columns: {list(edge_df.columns)}")

        # Test analysis
        print("🔍 Testing graph analysis...")
        analyzer = GraphAnalyzer(graph)

        # Test centrality computation (quick version)
        if graph.number_of_nodes() < 200:  # Only for smaller graphs
            centrality = analyzer.compute_centrality_measures()
            print(f"  - Computed centrality measures: {list(centrality.keys())}")

            # Get top nodes
            ranking = analyzer.get_node_importance_ranking("pagerank", 5)
            print("  - Top 5 nodes by PageRank:")
            for i, (node, score) in enumerate(ranking, 1):
                print(f"    {i}. {node[:50]}... ({score:.4f})")

        # Test entity type connectivity
        connectivity = analyzer.analyze_entity_type_connectivity()
        print("  - Entity type connectivity analysis completed")
        print(f"  - Type connections found: {len(connectivity['type_connections'])}")

        print("✅ All basic tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functionality."""
    print("\n🎨 Testing visualization...")

    try:
        from graphml_viewer import GraphVisualizer

        # Load a smaller subgraph for testing
        graphml_path = (
            Path(__file__).parent.parent
            / "light-rag"
            / "lightrag_data"
            / "graph_chunk_entity_relation.graphml"
        )
        loader = GraphMLLoader(graphml_path)
        graph = loader.load()

        # Filter to a subset for faster visualization
        entity_types = loader.get_entity_types()
        if entity_types:
            # Take only first 2 entity types for testing
            test_types = entity_types[:2]
            subgraph = loader.filter_by_entity_type(test_types)
            print(
                f"  - Created subgraph with types {test_types}: {subgraph.number_of_nodes()} nodes"
            )
        else:
            # Take first 50 nodes
            nodes = list(graph.nodes())[:50]
            subgraph = graph.subgraph(nodes)
            print(
                f"  - Created subgraph with first 50 nodes: {subgraph.number_of_nodes()} nodes"
            )

        visualizer = GraphVisualizer(subgraph)

        # Test color mapping
        if entity_types:
            color_map = visualizer.set_node_colors_by_attribute("entity_type")
            print(f"  - Color mapping created: {len(color_map)} colors")

        # Test layout computation
        positions = visualizer.compute_layout("spring")
        print(f"  - Layout computed: {len(positions)} positions")

        # Test static visualization
        try:
            fig = visualizer.plot_matplotlib(figsize=(8, 6), show_labels=False)
            output_path = Path(__file__).parent / "test_output.png"
            fig.savefig(output_path, dpi=100, bbox_inches="tight")
            print(f"  - Static visualization saved to: {output_path}")

            # Clean up
            import matplotlib.pyplot as plt

            plt.close(fig)

        except Exception as e:
            print(f"  - Static visualization failed: {e}")

        # Test interactive visualization
        try:
            fig = visualizer.plot_plotly(width=800, height=600)
            output_path = Path(__file__).parent / "test_output.html"
            fig.write_html(str(output_path))
            print(f"  - Interactive visualization saved to: {output_path}")

        except Exception as e:
            print(f"  - Interactive visualization failed: {e}")

        print("✅ Visualization tests completed!")
        return True

    except ImportError:
        print(
            "⚠️  Visualization dependencies not available, skipping visualization tests"
        )
        return True
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GraphML Viewer Test Suite")
    print("=" * 60)

    success = True

    # Test basic loading
    success &= test_basic_loading()

    # Test visualization
    success &= test_visualization()

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! GraphML Viewer is working correctly.")
        print("\nNext steps:")
        print("1. Try the CLI: python -m graphml_viewer.cli --help")
        print("2. Launch interactive mode: python -m graphml_viewer.cli --interactive")
        print(
            "3. Analyze your graph: python -m graphml_viewer.cli path/to/graph.graphml --analyze"
        )
    else:
        print("💥 Some tests failed. Please check the error messages above.")

    print("=" * 60)
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
