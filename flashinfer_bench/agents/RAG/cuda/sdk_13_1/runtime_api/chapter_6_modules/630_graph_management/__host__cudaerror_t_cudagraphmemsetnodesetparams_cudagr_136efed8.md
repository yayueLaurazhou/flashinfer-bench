# __host__cudaError_t cudaGraphMemsetNodeSetParams (cudaGraphNode_t node, const cudaMemsetParams *pNodeParams)

Sets a memset node's parameters.

##### Parameters

**node**

  - Node to set the parameters for
**pNodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the parameters of memset node node to pNodeParams.











See also:

cudaGraphNodeSetParams, cudaMemset2D, cudaGraphAddMemsetNode,
cudaGraphMemsetNodeGetParams


CUDA Runtime API vRelease Version  |  413


Modules