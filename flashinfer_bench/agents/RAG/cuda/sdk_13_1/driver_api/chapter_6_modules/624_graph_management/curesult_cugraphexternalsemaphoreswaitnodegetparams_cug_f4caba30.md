# CUresult cuGraphExternalSemaphoresWaitNodeGetParams (CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out)

Returns an external semaphore wait node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**params_out**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of an external semaphore wait node hNode in params_out. The
extSemArray and paramsArray returned in params_out, are owned by the node. This memory


CUDA Driver API TRM-06703-001 _vRelease Version  |  464


Modules


remains valid until the node is destroyed or its parameters are modified, and should not be modified
directly. Use cuGraphExternalSemaphoresSignalNodeSetParams to update the parameters of this node.





See also:

cuLaunchKernel, cuGraphAddExternalSemaphoresWaitNode,
cuGraphExternalSemaphoresWaitNodeSetParams, cuGraphAddExternalSemaphoresWaitNode,
cuSignalExternalSemaphoresAsync, cuWaitExternalSemaphoresAsync