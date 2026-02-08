# CUresult cuGraphMemFreeNodeGetParams (CUgraphNode hNode, CUdeviceptr *dptr_out)

Returns a memory free node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**dptr_out**

  - Pointer to return the device address

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the address of a memory free node hNode in dptr_out.





See also:

cuGraphAddMemFreeNode, cuGraphMemAllocNodeGetParams


CUDA Driver API TRM-06703-001 _vRelease Version  |  481


Modules