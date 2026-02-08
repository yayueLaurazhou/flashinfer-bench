# __host__cudaError_t cudaGraphExecNodeSetParams (cudaGraphExec_t graphExec, cudaGraphNode_t node, cudaGraphNodeParams *nodeParams)

Update's a graph node's parameters in an instantiated graph.

##### Parameters

**graphExec**

  - The executable graph in which to update the specified node
**node**

  - Corresponding node from the graph from which graphExec was instantiated
**nodeParams**

  - Updated Parameters to set

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction, cudaErrorNotSupported

##### Description

Sets the parameters of a node in an executable graph graphExec. The node is identified by
the corresponding node node in the non-executable graph from which the executable graph was
instantiated. node must not have been removed from the original graph.


CUDA Runtime API vRelease Version  |  375


Modules


The modifications only affect future launches of graphExec. Already enqueued or running launches
of graphExec are not affected by this call. node is also not modified by this call.

Allowed changes to parameters on executable graphs are as follows:











See also:

cudaGraphAddNode, cudaGraphNodeSetParams cudaGraphExecUpdate, cudaGraphInstantiate


CUDA Runtime API vRelease Version  |  376


Modules