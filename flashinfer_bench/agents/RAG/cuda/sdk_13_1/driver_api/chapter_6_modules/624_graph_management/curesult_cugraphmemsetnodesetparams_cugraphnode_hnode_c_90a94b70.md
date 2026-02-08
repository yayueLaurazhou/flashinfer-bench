# CUresult cuGraphMemsetNodeSetParams (CUgraphNode hNode, const CUDA_MEMSET_NODE_PARAMS *nodeParams)

Sets a memset node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy


CUDA Driver API TRM-06703-001 _vRelease Version  |  482


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Sets the parameters of memset node hNode to nodeParams.





See also:

cuGraphNodeSetParams, cuMemsetD2D32, cuGraphAddMemsetNode,
cuGraphMemsetNodeGetParams