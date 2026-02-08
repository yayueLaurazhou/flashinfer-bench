# CUresult cuGraphExternalSemaphoresSignalNodeSetParams (CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams)

Sets an external semaphore signal node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy


CUDA Driver API TRM-06703-001 _vRelease Version  |  463


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the parameters of an external semaphore signal node hNode to nodeParams.





See also:

cuGraphNodeSetParams, cuGraphAddExternalSemaphoresSignalNode,
cuGraphExternalSemaphoresSignalNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync