# enum CUgraphNodeType

Graph node types

###### Values

**CU_GRAPH_NODE_TYPE_KERNEL = 0**
GPU kernel node
**CU_GRAPH_NODE_TYPE_MEMCPY = 1**
Memcpy node
**CU_GRAPH_NODE_TYPE_MEMSET = 2**
Memset node
**CU_GRAPH_NODE_TYPE_HOST = 3**
Host (executable) node
**CU_GRAPH_NODE_TYPE_GRAPH = 4**
Node which executes an embedded graph
**CU_GRAPH_NODE_TYPE_EMPTY = 5**
Empty (no-op) node
**CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6**
External event wait node
**CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7**
External event record node
**CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8**
External semaphore signal node
**CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9**
External semaphore wait node
**CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10**
Memory Allocation Node
**CU_GRAPH_NODE_TYPE_MEM_FREE = 11**


CUDA Driver API TRM-06703-001 _vRelease Version  |  50


Modules


Memory Free Node
**CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12**
Batch MemOp Node See cuStreamBatchMemOp and CUstreamBatchMemOpType for what these
nodes can do.
**CU_GRAPH_NODE_TYPE_CONDITIONAL = 13**
Conditional NodeMay be used to implement a conditional execution path or loop inside of a graph.
The graph(s) contained within the body of the conditional node can be selectively executed or
iterated upon based on the value of a conditional variable.Handles must be created in advance of
creating the node using cuGraphConditionalHandleCreate.The following restrictions apply to graphs
which contain conditional nodes: The graph cannot be used in a child node. Only one instantiation
of the graph may exist at any point in time. The graph cannot be cloned.To set the control value,
supply a default value when creating the handle and/or call cudaGraphSetConditional from device
code.