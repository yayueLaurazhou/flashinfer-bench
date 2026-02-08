"""
Visualization tools for the CUDA knowledge graph.
Supports multiple visualization methods: Pyvis, Plotly, Matplotlib, and Gephi export.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import networkx as nx

# Visualization libraries
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphVisualizer:
    """Visualize knowledge graphs using multiple methods."""

    # Color scheme for entity categories
    CATEGORY_COLORS = {
        "Concept": "#FF6B6B",  # Red
        "Hardware": "#4ECDC4",  # Teal
        "SoftwareComponent": "#45B7D1",  # Blue
        "Function": "#FFA07A",  # Light Salmon
        "Keyword/Type": "#98D8C8",  # Mint
        "Metric": "#F7DC6F",  # Yellow
        "Tool": "#BB8FCE",  # Purple
    }

    # Relationship type colors
    RELATIONSHIP_COLORS = {
        "uses_principle": "#FF6B6B",
        "implements": "#4ECDC4",
        "executes_on": "#45B7D1",
        "measures": "#FFA07A",
        "applied_in": "#98D8C8",
    }

    def __init__(self, graph_path: Path):
        """
        Initialize the visualizer.

        Args:
            graph_path: Path to the saved graph pickle file
        """
        self.graph = self._load_graph(graph_path)
        if self.graph is None:
            raise ValueError(f"Could not load graph from {graph_path}")

    def _load_graph(self, graph_path: Path) -> Optional[nx.DiGraph]:
        """Load graph from pickle file."""
        pickle_path = Path(graph_path).with_suffix(".pkl")
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                graph = pickle.load(f)
            logger.info(f"Loaded graph from {pickle_path}")
            return graph
        else:
            logger.error(f"Graph file not found: {pickle_path}")
            return None

    def visualize_pyvis(self, output_file: str = "knowledge_graph.html"):
        """
        Visualize graph using Pyvis (interactive HTML).

        Args:
            output_file: Output HTML file path
        """
        if not HAS_PYVIS:
            logger.error("Pyvis not installed. Install with: pip install pyvis")
            return

        logger.info("Creating Pyvis visualization...")

        # Create network
        net = Network(
            height="750px",
            width="100%",
            directed=True,
            notebook=False,
            physics=True,
        )

        # Add nodes with colors and labels
        for node, data in self.graph.nodes(data=True):
            category = data.get("category", "Unknown")
            entity = data.get("entity", "Unknown")
            color = self.CATEGORY_COLORS.get(category, "#808080")

            net.add_node(
                node,
                label=entity,
                title=f"{category}: {entity}",
                color=color,
                size=20,
                font={"size": 14},
            )

        # Add edges with relationship labels
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("relationship_type", "related_to")
            edge_color = self.RELATIONSHIP_COLORS.get(rel_type, "#999999")

            net.add_edge(
                source,
                target,
                title=rel_type,
                label=rel_type,
                color=edge_color,
                arrows="to",
                font={"size": 12},
            )

        # Configure physics
        net.toggle_physics(True)
        net.show_buttons(filter_=["physics", "interaction", "selection"])
        net.show(output_file)

        logger.info(f"Pyvis visualization saved to {output_file}")

    def visualize_plotly(self, output_file: str = "knowledge_graph_plotly.html"):
        """
        Visualize graph using Plotly (interactive 2D/3D).

        Args:
            output_file: Output HTML file path
        """
        if not HAS_PLOTLY:
            logger.error("Plotly not installed. Install with: pip install plotly")
            return

        logger.info("Creating Plotly visualization...")

        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Extract node positions
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []

        for node, (x, y) in pos.items():
            node_x.append(x)
            node_y.append(y)

            data = self.graph.nodes[node]
            category = data.get("category", "Unknown")
            entity = data.get("entity", "Unknown")

            node_text.append(f"{category}: {entity}")
            node_colors.append(self.CATEGORY_COLORS.get(category, "#808080"))
            node_sizes.append(15)

        # Extract edges
        edge_x = []
        edge_y = []
        edge_colors = []
        edge_labels = []

        for source, target, data in self.graph.edges(data=True):
            if source in pos and target in pos:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                rel_type = data.get("relationship_type", "related_to")
                edge_colors.extend(
                    [
                        self.RELATIONSHIP_COLORS.get(rel_type, "#999999"),
                        self.RELATIONSHIP_COLORS.get(rel_type, "#999999"),
                        None,
                    ]
                )
                edge_labels.append(rel_type)

        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            showlegend=False,
        )

        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[nt.split(":")[1] for nt in node_text],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line_width=2,
                line_color="white",
            ),
            showlegend=False,
        )

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace])

        fig.update_layout(
            title="CUDA Knowledge Graph",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=800,
            width=1200,
        )

        fig.write_html(output_file)
        logger.info(f"Plotly visualization saved to {output_file}")

    def visualize_matplotlib(self, output_file: str = "knowledge_graph_matplotlib.png"):
        """
        Visualize graph using Matplotlib (static image).

        Args:
            output_file: Output image file path
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not installed. Install with: pip install matplotlib")
            return

        logger.info("Creating Matplotlib visualization...")

        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Use spring layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            edge_color="gray",
            arrows=True,
            arrowsize=10,
            arrowstyle="->",
            width=0.5,
            alpha=0.5,
        )

        # Group nodes by category
        nodes_by_category = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            category = data.get("category", "Unknown")
            nodes_by_category[category].append(node)

        # Draw nodes by category
        for category, nodes in nodes_by_category.items():
            node_positions = {n: pos[n] for n in nodes}
            color = self.CATEGORY_COLORS.get(category, "#808080")

            nx.draw_networkx_nodes(
                self.graph.subgraph(nodes),
                node_positions,
                ax=ax,
                node_color=color,
                node_size=300,
                label=category,
                alpha=0.9,
                edgecolors="black",
                linewidths=1.5,
            )

        # Draw labels (only for high-degree nodes to avoid clutter)
        high_degree_nodes = [n for n, d in self.graph.degree() if d > 2]
        labels = {n: self.graph.nodes[n].get("entity", n) for n in high_degree_nodes}

        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels,
            font_size=8,
            font_weight="bold",
            ax=ax,
        )

        ax.set_title("CUDA Knowledge Graph", fontsize=16, fontweight="bold")
        ax.legend(scatterpoints=1, loc="upper left", fontsize=10)
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        logger.info(f"Matplotlib visualization saved to {output_file}")
        plt.close()

    def visualize_subgraph(
        self,
        entity: str,
        category: str,
        depth: int = 1,
        output_file: str = "subgraph.html",
    ):
        """
        Visualize a subgraph around a specific entity.

        Args:
            entity: Entity name
            category: Entity category
            depth: Depth of neighbors to include
            output_file: Output HTML file path
        """
        node_id = f"{category}:{entity}"

        if node_id not in self.graph:
            logger.error(f"Entity not found: {node_id}")
            return

        # Extract subgraph
        nodes = {node_id}
        for _ in range(depth):
            new_nodes = set()
            for node in nodes:
                new_nodes.update(self.graph.predecessors(node))
                new_nodes.update(self.graph.successors(node))
            nodes.update(new_nodes)

        subgraph = self.graph.subgraph(nodes)
        logger.info(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

        # Visualize using Pyvis
        if HAS_PYVIS:
            net = Network(height="750px", width="100%", directed=True)

            for node, data in subgraph.nodes(data=True):
                cat = data.get("category", "Unknown")
                ent = data.get("entity", "Unknown")
                color = self.CATEGORY_COLORS.get(cat, "#808080")
                size = 25 if node == node_id else 15

                net.add_node(
                    node,
                    label=ent,
                    title=f"{cat}: {ent}",
                    color=color,
                    size=size,
                    font={"size": 14},
                )

            for source, target, data in subgraph.edges(data=True):
                rel_type = data.get("relationship_type", "related_to")
                edge_color = self.RELATIONSHIP_COLORS.get(rel_type, "#999999")

                net.add_edge(
                    source,
                    target,
                    title=rel_type,
                    label=rel_type,
                    color=edge_color,
                    arrows="to",
                )

            net.show(output_file)
            logger.info(f"Subgraph visualization saved to {output_file}")
        else:
            logger.error("Pyvis required for subgraph visualization")

    def export_to_gephi(self, output_file: str = "knowledge_graph.gexf"):
        """
        Export graph to GEXF format (readable by Gephi).

        Args:
            output_file: Output GEXF file path
        """
        logger.info(f"Exporting graph to GEXF format...")

        # Add visualization attributes
        for node, data in self.graph.nodes(data=True):
            category = data.get("category", "Unknown")
            data["color"] = self.CATEGORY_COLORS.get(category, "#808080")
            data["size"] = 10

        nx.write_gexf(self.graph, output_file)
        logger.info(f"Graph exported to {output_file}")
        logger.info("Open this file with Gephi for advanced visualization")

    def export_to_graphml(self, output_file: str = "knowledge_graph.graphml"):
        """
        Export graph to GraphML format.

        Args:
            output_file: Output GraphML file path
        """
        logger.info(f"Exporting graph to GraphML format...")
        nx.write_graphml(self.graph, output_file)
        logger.info(f"Graph exported to {output_file}")

    def print_graph_info(self):
        """Print information about the graph."""
        logger.info(f"Graph Statistics:")
        logger.info(f"  Nodes: {self.graph.number_of_nodes()}")
        logger.info(f"  Edges: {self.graph.number_of_edges()}")
        logger.info(f"  Density: {nx.density(self.graph):.4f}")

        nodes_by_category = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            category = data.get("category", "Unknown")
            nodes_by_category[category] += 1

        logger.info(f"  Nodes by category:")
        for category, count in sorted(nodes_by_category.items()):
            logger.info(f"    {category}: {count}")

        edges_by_type = defaultdict(int)
        for source, target, data in self.graph.edges(data=True):
            rel_type = data.get("relationship_type", "Unknown")
            edges_by_type[rel_type] += 1

        logger.info(f"  Edges by type:")
        for rel_type, count in sorted(edges_by_type.items()):
            logger.info(f"    {rel_type}: {count}")

        # Top nodes by degree
        top_nodes = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:10]
        logger.info(f"  Top 10 connected entities:")
        for node, degree in top_nodes:
            entity = self.graph.nodes[node].get("entity", node)
            logger.info(f"    {entity}: {degree}")


def main():
    """Main function to visualize the graph."""
    # Get the knowledge base path
    current_dir = Path(__file__).parent
    graph_path = current_dir / "knowledge_base" / "cuda_knowledge_graph.pkl"

    if not graph_path.exists():
        logger.error(f"Graph file not found at {graph_path}")
        logger.info("Please run build_graph2.py first to generate the graph")
        return

    # Create visualizer
    visualizer = GraphVisualizer(graph_path)

    # Print graph info
    visualizer.print_graph_info()

    # Create visualizations
    output_dir = current_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating visualizations...")

    # Pyvis visualization (interactive, best for exploration)
    if HAS_PYVIS:
        visualizer.visualize_pyvis(output_dir / "knowledge_graph.html")
    else:
        logger.warning("Install pyvis for interactive visualization: pip install pyvis")

    # Plotly visualization (interactive with zoom/pan)
    if HAS_PLOTLY:
        visualizer.visualize_plotly(output_dir / "knowledge_graph_plotly.html")
    else:
        logger.warning("Install plotly for 2D/3D visualization: pip install plotly")

    # Matplotlib visualization (static image)
    if HAS_MATPLOTLIB:
        visualizer.visualize_matplotlib(output_dir / "knowledge_graph_matplotlib.png")
    else:
        logger.warning("Install matplotlib for static visualization: pip install matplotlib")

    # Export formats
    visualizer.export_to_gexf(output_dir / "knowledge_graph.gexf")
    visualizer.export_to_graphml(output_dir / "knowledge_graph.graphml")

    # Subgraph example: visualize around "occupancy"
    logger.info("\nGenerating subgraph visualization...")
    visualizer.visualize_subgraph("occupancy", "Concept", depth=1, output_file=str(output_dir / "subgraph_occupancy.html"))

    logger.info(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
