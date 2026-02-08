# __host__cudaError_t cudaGraphMemcpyNodeGetParams (cudaGraphNode_t node, cudaMemcpy3DParms *pNodeParams)

Returns a memcpy node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**pNodeParams**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the parameters of memcpy node node in pNodeParams.











See also:

cudaMemcpy3D, cudaGraphAddMemcpyNode, cudaGraphMemcpyNodeSetParams


CUDA Runtime API vRelease Version  |  406


Modules