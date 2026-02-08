# __host__cudaError_t cudaGraphNodeSetEnabled (cudaGraphExec_t hGraphExec, cudaGraphNode_t hNode, unsigned int isEnabled)

Enables or disables the specified node in the given graphExec.

##### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Node from the graph from which graphExec was instantiated
**isEnabled**

  - Node is enabled if != 0, otherwise the node is disabled


CUDA Runtime API vRelease Version  |  420


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets hNode to be either enabled or disabled. Disabled nodes are functionally equivalent to empty
nodes until they are reenabled. Existing node parameters are not affected by disabling/enabling the
node.

The node is identified by the corresponding node hNode in the non-executable graph, from which the
executable graph was instantiated.

hNode must not have been removed from the original graph.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.


Note:


Currently only kernel, memset and memcpy nodes are supported.











See also:

cudaGraphNodeGetEnabled, cudaGraphExecUpdate, cudaGraphInstantiate cudaGraphLaunch