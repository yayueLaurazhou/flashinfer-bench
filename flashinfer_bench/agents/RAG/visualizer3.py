import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# Load the graph
G = nx.read_graphml("/home/zhouyayue/flashinfer-bench/flashinfer_bench/agents/RAG/knowledge_base/cuda_knowledge_graph.graphml")

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Count by category
categories = Counter(data.get("category", "unknown") for _, data in G.nodes(data=True))
print("\nNodes by category:")
for cat, count in categories.most_common():
    print(f"  {cat}: {count}")

# --- Option A: Visualize a subgraph (recommended for large graphs) ---
# Filter to only "Concept" nodes for a manageable view
concept_nodes = [n for n, d in G.nodes(data=True) if d.get("category") == "Concept"]
subgraph = G.subgraph(concept_nodes[:50])  # Take first 50

plt.figure(figsize=(20, 15))
pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
nx.draw(
    subgraph,
    pos,
    with_labels=True,
    node_size=300,
    font_size=6,
    node_color="skyblue",
    edge_color="gray",
    alpha=0.7,
    arrows=True,
)
plt.title("CUDA Knowledge Graph - Concept Nodes (subset)")
plt.tight_layout()
plt.savefig("graph_concepts.png", dpi=150)
plt.show()