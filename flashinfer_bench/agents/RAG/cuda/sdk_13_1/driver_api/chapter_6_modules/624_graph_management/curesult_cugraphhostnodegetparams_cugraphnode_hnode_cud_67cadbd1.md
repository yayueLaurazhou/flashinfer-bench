# CUresult cuGraphHostNodeGetParams (CUgraphNode hNode, CUDA_HOST_NODE_PARAMS *nodeParams)

Returns a host node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**nodeParams**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of host node hNode in nodeParams.





See also:

cuLaunchHostFunc, cuGraphAddHostNode, cuGraphHostNodeSetParams


CUDA Driver API TRM-06703-001 _vRelease Version  |  469


Modules