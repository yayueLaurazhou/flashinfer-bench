# unsigned char CUgraphEdgeData::from_port

This indicates when the dependency is triggered from the upstream node on the edge.
The meaning is specfic to the node type. A value of 0 in all cases means full completion
of the upstream node, with memory visibility to the downstream node or portion thereof
(indicated by to_port). Only kernel nodes define non-zero ports. A kernel node can
use the following output port types: CU_GRAPH_KERNEL_NODE_PORT_DEFAULT,
CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC, or
CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER.


CUDA Driver API TRM-06703-001 _vRelease Version  |  719


Data Structures