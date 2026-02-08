# CUgraph CUDA_CHILD_GRAPH_NODE_PARAMS::graph

The child graph to clone into the node for node creation, or a handle to the graph owned by the node
for node query. The graph must not contain conditional nodes. Graphs containing memory allocation or
memory free nodes must set the ownership to be moved to the parent.