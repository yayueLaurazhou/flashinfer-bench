# CUresult cuGraphMemcpyNodeSetParams (CUgraphNode hNode, const CUDA_MEMCPY3D *nodeParams)

Sets a memcpy node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  480


Modules

###### Description

Sets the parameters of memcpy node hNode to nodeParams.





See also:

cuGraphNodeSetParams, cuMemcpy3D, cuGraphAddMemcpyNode, cuGraphMemcpyNodeGetParams