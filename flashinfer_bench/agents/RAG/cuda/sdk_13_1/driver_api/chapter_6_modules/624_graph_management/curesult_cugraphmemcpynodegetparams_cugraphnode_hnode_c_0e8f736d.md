# CUresult cuGraphMemcpyNodeGetParams (CUgraphNode hNode, CUDA_MEMCPY3D *nodeParams)

Returns a memcpy node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**nodeParams**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of memcpy node hNode in nodeParams.





See also:

cuMemcpy3D, cuGraphAddMemcpyNode, cuGraphMemcpyNodeSetParams