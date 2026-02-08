# __host__cudaError_t cudaGraphHostNodeGetParams (cudaGraphNode_t node, cudaHostNodeParams *pNodeParams)

Returns a host node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**pNodeParams**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the parameters of host node node in pNodeParams.











See also:

cudaLaunchHostFunc, cudaGraphAddHostNode, cudaGraphHostNodeSetParams


CUDA Runtime API vRelease Version  |  387


Modules