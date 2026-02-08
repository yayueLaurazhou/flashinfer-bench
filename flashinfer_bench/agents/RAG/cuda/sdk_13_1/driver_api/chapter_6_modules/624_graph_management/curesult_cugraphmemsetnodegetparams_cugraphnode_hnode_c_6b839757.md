# CUresult cuGraphMemsetNodeGetParams (CUgraphNode hNode, CUDA_MEMSET_NODE_PARAMS *nodeParams)

Returns a memset node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**nodeParams**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of memset node hNode in nodeParams.





See also:

cuMemsetD2D32, cuGraphAddMemsetNode, cuGraphMemsetNodeSetParams