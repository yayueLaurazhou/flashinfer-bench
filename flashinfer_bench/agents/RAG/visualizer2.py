# pip install pyvis
import networkx as nx
from pyvis.network import Network

G = nx.read_graphml("/home/zhouyayue/flashinfer-bench/flashinfer_bench/agents/RAG/knowledge_base/cuda_knowledge_graph.graphml")

# Color map by category
color_map = {
    "Concept": "#4CAF50",
    "Hardware": "#2196F3",
    "SoftwareComponent": "#FF9800",
    "Function": "#9C27B0",
    "Keyword/Type": "#F44336",
    "Metric": "#00BCD4",
    "Tool": "#795548",
}

# Filter to a manageable subset (top connected nodes)
degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
top_nodes = [n for n, d in degree_sorted[:200]]
subgraph = G.subgraph(top_nodes)

net = Network(height="900px", width="100%", directed=True, notebook=False)
net.barnes_hut(gravity=-3000, spring_length=200)

for node, data in subgraph.nodes(data=True):
    category = data.get("category", "unknown")
    label = data.get("entity", node)
    # Truncate long labels
    if len(label) > 40:
        label = label[:40] + "..."
    net.add_node(
        node,
        label=label,
        color=color_map.get(category, "#999999"),
        title=f"Category: {category}\nEntity: {data.get('entity', '')}",
        size=10 + subgraph.degree(node) * 2,
    )

for u, v, data in subgraph.edges(data=True):
    rel = data.get("relationship_type", "")
    net.add_edge(u, v, title=rel, label=rel, font={"size": 8})

net.show("cuda_knowledge_graph.html", notebook=False)
print("Open cuda_knowledge_graph.html in your browser")