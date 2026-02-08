# enum CUgraphDependencyType

Type annotations that can be applied to graph edges as part of CUgraphEdgeData.

###### Values

**CU_GRAPH_DEPENDENCY_TYPE_DEFAULT = 0**
This is an ordinary dependency.
**CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC = 1**
This dependency type allows the downstream node to use
cudaGridDependencySynchronize() . It may only be used between kernel nodes, and
must be used with either the CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC or
CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER outgoing port.