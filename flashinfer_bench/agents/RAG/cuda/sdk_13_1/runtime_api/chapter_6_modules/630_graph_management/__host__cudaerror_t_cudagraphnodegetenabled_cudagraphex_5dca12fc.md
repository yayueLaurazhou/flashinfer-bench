# __host__cudaError_t cudaGraphNodeGetEnabled (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int *isEnabled)

Query whether a node in the given graphExec is enabled.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node


CUDA Runtime API vRelease Version  |  417


Modules


**hNode**

  - Node from the graph from which graphExec was instantiated
**isEnabled**

  - Location to return the enabled status of the node

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets isEnabled to 1 if hNode is enabled, or 0 if hNode is disabled.

The node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.


Note:


Currently only kernel, memset and memcpy nodes are supported.











See also:

cudaGraphNodeSetEnabled, cudaGraphExecUpdate, cudaGraphInstantiate cudaGraphLaunch