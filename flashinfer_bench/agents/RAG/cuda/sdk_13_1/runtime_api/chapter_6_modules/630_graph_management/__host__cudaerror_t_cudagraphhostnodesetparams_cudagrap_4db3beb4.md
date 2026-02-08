# __host__cudaError_t cudaGraphHostNodeSetParams (cudaGraphNode_t node, const cudaHostNodeParams *pNodeParams)

Sets a host node's parameters.

##### Parameters

**node**

  - Node to set the parameters for
**pNodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Sets the parameters of host node node to nodeParams.











See also:

cudaGraphNodeSetParams, cudaLaunchHostFunc, cudaGraphAddHostNode,
cudaGraphHostNodeGetParams


CUDA Runtime API vRelease Version  |  388


Modules