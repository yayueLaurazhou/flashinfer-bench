# __host__cudaError_t cudaGraphExternalSemaphoresSignalNodeSetParams (cudaGraphNode_t hNode, const cudaExternalSemaphoreSignalNodeParams *nodeParams)

Sets an external semaphore signal node's parameters.

##### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy


CUDA Runtime API vRelease Version  |  380


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the parameters of an external semaphore signal node hNode to nodeParams.











See also:

cudaGraphNodeSetParams, cudaGraphAddExternalSemaphoresSignalNode,
cudaGraphExternalSemaphoresSignalNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync