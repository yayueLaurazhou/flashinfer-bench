# CUresult cuGraphExecMemsetNodeSetParams (CUgraphExec hGraphExec, CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *memsetParams, CUcontext ctx)

Sets the parameters for a memset node in the given graphExec.

###### Parameters

**hGraphExec**

  - The executable graph in which to set the specified node
**hNode**

  - Memset node from the graph which was used to instantiate graphExec
**memsetParams**

  - The updated parameters to set
**ctx**

  - Context on which to run the node

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE,

###### Description

Updates the work represented by hNode in hGraphExec as though hNode had contained
memsetParams at instantiation. hNode must remain in the graph which was used to instantiate
hGraphExec. Changed edges to and from hNode are ignored.

Zero sized operations are not supported.

The new destination pointer in memsetParams must be to the same kind of allocation as the original
destination pointer and have the same context association and device mapping as the original
destination pointer.

Both the value and pointer address may be updated. Changing other aspects of the memset (width,
height, element size or pitch) may cause the update to be rejected. Specifically, for 2d memsets, all
dimension changes are rejected. For 1d memsets, changes in height are explicitly rejected and other
changes are opportunistically allowed if the resulting work maps onto the work resources already
allocated for the node.

The modifications only affect future launches of hGraphExec. Already enqueued or running
launches of hGraphExec are not affected by this call. hNode is also not modified by this call.


CUDA Driver API TRM-06703-001 _vRelease Version  |  457


Modules





See also:

cuGraphExecNodeSetParams, cuGraphAddMemsetNode, cuGraphMemsetNodeSetParams,
cuGraphExecKernelNodeSetParams, cuGraphExecMemcpyNodeSetParams,
cuGraphExecHostNodeSetParams, cuGraphExecChildGraphNodeSetParams,
cuGraphExecEventRecordNodeSetEvent, cuGraphExecEventWaitNodeSetEvent,
cuGraphExecExternalSemaphoresSignalNodeSetParams,
cuGraphExecExternalSemaphoresWaitNodeSetParams, cuGraphExecUpdate, cuGraphInstantiate