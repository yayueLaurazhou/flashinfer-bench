# CUresult cuGraphMemAllocNodeGetParams (CUgraphNode hNode, CUDA_MEM_ALLOC_NODE_PARAMS *params_out)

Returns a memory alloc node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**params_out**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of a memory alloc node hNode in params_out. The poolProps and
accessDescs returned in params_out, are owned by the node. This memory remains valid until
the node is destroyed. The returned parameters must not be modified.





See also:

cuGraphAddMemAllocNode, cuGraphMemFreeNodeGetParams


CUDA Driver API TRM-06703-001 _vRelease Version  |  479


Modules