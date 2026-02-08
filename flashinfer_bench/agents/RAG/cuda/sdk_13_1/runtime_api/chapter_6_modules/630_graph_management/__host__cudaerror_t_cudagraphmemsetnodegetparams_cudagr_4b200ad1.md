# __host__cudaError_t cudaGraphMemsetNodeGetParams (cudaGraphNode_t node, cudaMemsetParams *pNodeParams)

Returns a memset node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**pNodeParams**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the parameters of memset node node in pNodeParams.









CUDA Runtime API vRelease Version  |  412


Modules





See also:

cudaMemset2D, cudaGraphAddMemsetNode, cudaGraphMemsetNodeSetParams