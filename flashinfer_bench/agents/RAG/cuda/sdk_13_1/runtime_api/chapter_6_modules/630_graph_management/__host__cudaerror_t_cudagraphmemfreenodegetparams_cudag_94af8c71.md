# __host__cudaError_t cudaGraphMemFreeNodeGetParams (cudaGraphNode_t node, void *dptr_out)

Returns a memory free node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**dptr_out**

  - Pointer to return the device address

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  411


Modules

##### Description

Returns the address of a memory free node hNode in dptr_out.











See also:

cudaGraphAddMemFreeNode, cudaGraphMemFreeNodeGetParams