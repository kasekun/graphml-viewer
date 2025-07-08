# GraphML Viewer

A tool for visualizing and exploring LightRAG knowledge graphs stored in GraphML format.

## Features

- 📊 **Interactive Graph Visualization** - Explore knowledge graphs with interactive plots
- 🔍 **Search & Filter** - Find nodes and explore neighborhoods  
- 📈 **Graph Analytics** - Centrality measures, community detection, connectivity analysis
- 🎨 **Customizable Styling** - Color nodes by type, multiple layout algorithms
- 🖥️ **Multiple Interfaces** - Command-line tool and web interface
- 📋 **Export** - Save visualizations and data tables

## Installation

```bash
# Clone or navigate to the graphml-viewer directory
cd graphml-viewer

# Install with UV
uv sync

# Test the installation
uv run python tests/test_basic.py
```

## Quick Start

### Web Interface (Recommended)

```bash
# Launch interactive web interface
uv run python -m graphml_viewer.cli --interactive
```

This opens a Streamlit app with tabs for visualization, exploration, analytics, and data export.

### Command Line Usage

```bash
# Show help
uv run python -m graphml_viewer.cli --help

# Analyze a GraphML file
uv run python -m graphml_viewer.cli path/to/graph.graphml --analyze

# Create static visualization
uv run python -m graphml_viewer.cli path/to/graph.graphml --output visualization.png

# Create interactive HTML visualization  
uv run python -m graphml_viewer.cli path/to/graph.graphml --output visualization.html --layout spring

# Filter by entity types
uv run python -m graphml_viewer.cli path/to/graph.graphml --filter-types "organization" "event" --output filtered.html
```

### Python API

```python
from graphml_viewer import GraphMLLoader, GraphVisualizer, GraphAnalyzer

# Load and analyze
loader = GraphMLLoader("path/to/graph.graphml")
graph = loader.load()
stats = loader.get_graph_statistics()

# Visualize
visualizer = GraphVisualizer(graph)
visualizer.set_node_colors_by_attribute("entity_type")
fig = visualizer.plot_plotly(layout="spring")
fig.show()

# Analyze
analyzer = GraphAnalyzer(graph)
centrality = analyzer.compute_centrality_measures()
top_nodes = analyzer.get_node_importance_ranking("pagerank", 10)
```

## GraphML Format

Designed for [LightRAG](https://github.com/HKUDS/LightRAG) knowledge graphs with these attributes:

**Nodes**: `entity_id`, `entity_type`, `description`, `source_id`  
**Edges**: `weight`, `description`, `keywords`, `source_id`

## Layout Options

- `spring` - Force-directed (default)
- `circular` - Circular arrangement  
- `kamada_kawai` - Force-directed with better aesthetics
- `spectral` - Based on graph spectrum
- `random` - Random positions

## Analytics

**Centrality Measures**: Degree, Betweenness, Closeness, Eigenvector, PageRank  
**Community Detection**: Louvain, Greedy Modularity, Label Propagation  
**Connectivity**: Entity type relationship patterns

## Output Formats

- **Static**: PNG, PDF, SVG (matplotlib)
- **Interactive**: HTML (plotly) 
- **Data**: CSV export for nodes/edges

## Performance Tips

- Filter large graphs by entity type: `--filter-types "organization"`
- Interactive visualizations work best with <500 nodes
- Use static plots for very large graphs
- Sample or extract subgraphs for detailed analysis

## Troubleshooting

**Large graphs**: Use filtering or switch to static visualization  
**Layout issues**: Try different layout algorithms  
**Import errors**: Ensure all dependencies installed with `uv sync`

## Requirements

- Python 3.12+
- NetworkX, Pandas, Matplotlib, Plotly, Streamlit

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Code formatting
uv run ruff check .
uv run ruff format .
```

