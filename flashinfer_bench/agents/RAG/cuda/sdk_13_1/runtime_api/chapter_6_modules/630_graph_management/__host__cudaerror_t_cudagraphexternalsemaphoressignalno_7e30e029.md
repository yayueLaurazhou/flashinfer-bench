# __host__cudaError_t cudaGraphExternalSemaphoresSignalNodeGetParams (cudaGraphNode_t hNode, cudaExternalSemaphoreSignalNodeParams *params_out)

Returns an external semaphore signal node's parameters.

##### Parameters

**hNode**

  - Node to get the parameters for


CUDA Runtime API vRelease Version  |  379


Modules


**params_out**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the parameters of an external semaphore signal node hNode in params_out. The
extSemArray and paramsArray returned in params_out, are owned by the node. This memory
remains valid until the node is destroyed or its parameters are modified, and should not be modified
directly. Use cudaGraphExternalSemaphoresSignalNodeSetParams to update the parameters of this
node.











See also:

cudaLaunchKernel, cudaGraphAddExternalSemaphoresSignalNode,
cudaGraphExternalSemaphoresSignalNodeSetParams, cudaGraphAddExternalSemaphoresWaitNode,
cudaSignalExternalSemaphoresAsync, cudaWaitExternalSemaphoresAsync