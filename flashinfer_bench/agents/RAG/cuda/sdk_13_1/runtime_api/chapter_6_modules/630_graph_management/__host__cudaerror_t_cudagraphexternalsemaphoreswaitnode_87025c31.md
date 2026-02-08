# __host__cudaError_t cudaGraphExternalSemaphoresWaitNodeGetParams (cudaGraphNode_t hNode, cudaExternalSemaphoreWaitNodeParams *params_out)

Returns an external semaphore wait node's parameters.

##### Parameters

**hNode**

  - Node to get the parameters for
**params_out**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  381


Modules

##### Description

Returns the parameters of an external semaphore wait node hNode in params_out. The
extSemArray and paramsArray returned in params_out, are owned by the node. This memory
remains valid until the node is destroyed or its parameters are modified, and should not be modified
directly. Use cudaGraphExternalSemaphoresSignalNodeSetParams to update the parameters of this
node.











See also:

cudaLaunchKernel, cudaGraphAddExternalSemaphoresWaitNode,
cudaGraphExternalSemaphoresWaitNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync