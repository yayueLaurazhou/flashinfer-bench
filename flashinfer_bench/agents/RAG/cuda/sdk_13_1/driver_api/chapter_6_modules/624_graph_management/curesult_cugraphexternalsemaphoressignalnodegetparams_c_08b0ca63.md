# CUresult cuGraphExternalSemaphoresSignalNodeGetParams (CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out)

Returns an external semaphore signal node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for


CUDA Driver API TRM-06703-001 _vRelease Version  |  462


Modules


**params_out**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of an external semaphore signal node hNode in params_out. The
extSemArray and paramsArray returned in params_out, are owned by the node. This memory
remains valid until the node is destroyed or its parameters are modified, and should not be modified
directly. Use cuGraphExternalSemaphoresSignalNodeSetParams to update the parameters of this node.





See also:

cuLaunchKernel, cuGraphAddExternalSemaphoresSignalNode,
cuGraphExternalSemaphoresSignalNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync