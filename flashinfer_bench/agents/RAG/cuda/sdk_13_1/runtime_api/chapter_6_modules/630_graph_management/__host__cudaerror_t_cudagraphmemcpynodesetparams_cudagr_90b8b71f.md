# __host__cudaError_t cudaGraphMemcpyNodeSetParams (cudaGraphNode_t node, const cudaMemcpy3DParms *pNodeParams)

Sets a memcpy node's parameters.

##### Parameters

**node**

  - Node to set the parameters for
**pNodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue,

##### Description

Sets the parameters of memcpy node node to pNodeParams.











See also:

cudaGraphNodeSetParams, cudaMemcpy3D, cudaGraphMemcpyNodeSetParamsToSymbol,
cudaGraphMemcpyNodeSetParamsFromSymbol, cudaGraphMemcpyNodeSetParams1D,
cudaGraphAddMemcpyNode, cudaGraphMemcpyNodeGetParams


CUDA Runtime API vRelease Version  |  407


Modules