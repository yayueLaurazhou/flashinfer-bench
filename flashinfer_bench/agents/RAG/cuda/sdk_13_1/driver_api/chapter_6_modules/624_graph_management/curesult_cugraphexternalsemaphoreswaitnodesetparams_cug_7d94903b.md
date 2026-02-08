# CUresult cuGraphExternalSemaphoresWaitNodeSetParams (CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams)

Sets an external semaphore wait node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the parameters of an external semaphore wait node hNode to nodeParams.





See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  465


Modules


cuGraphNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuGraphExternalSemaphoresWaitNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync