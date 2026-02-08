# __host__cudaError_t cudaGraphExternalSemaphoresWaitNodeSetParams (cudaGraphNode_t hNode, const cudaExternalSemaphoreWaitNodeParams *nodeParams)

Sets an external semaphore wait node's parameters.

##### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  382


Modules

##### Description

Sets the parameters of an external semaphore wait node hNode to nodeParams.











See also:

cudaGraphNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaGraphExternalSemaphoresWaitNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync