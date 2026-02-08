# CUresult cuGraphExecMemcpyNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMCPY3D *copyParams, CUcontext ctx)

Sets the parameters for a memcpy node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Memcpy node from the graph which was used to instantiate graphExec
**copyParams**

  - The updated parameters to set
**ctx**

  - Context on which to run the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Updates the work represented by hNode in hGraphExec as though hNode had contained
copyParams at instantiation. hNode must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from hNode are ignored.

The source and destination memory in copyParams must be allocated from the same contexts as the
original source and destination memory. Both the instantiation-time memory operands and the memory
operands in copyParams must be 1-dimensional. Zero-length operations are not supported.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.

Returns CUDA_ERROR_INVALID_VALUE if the memory operands' mappings changed or either the
original or new memory operands are multidimensional.





See also:

cuGraphExecNodeSetParams, cuGraphAddMemcpyNode,
cuGraphMemcpyNodeSetParams, cuGraphExecKernelNodeSetParams,


CUDA Driver API TRM-06703-001 _vRelease Version  |  456


Modules


cuGraphExecMemsetNodeSetParams, cuGraphExecHostNodeSetParams,
cuGraphExecChildGraphNodeSetParams, cuGraphExecEventRecordNodeSetEvent,
cuGraphExecEventWaitNodeSetEvent, cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate