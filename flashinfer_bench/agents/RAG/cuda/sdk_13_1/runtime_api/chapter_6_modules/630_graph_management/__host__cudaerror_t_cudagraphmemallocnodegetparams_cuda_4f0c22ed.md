# __host__cudaError_t cudaGraphMemAllocNodeGetParams (cudaGraphNode_t node, cudaMemAllocNodeParams *params_out)

Returns a memory alloc node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**params_out**

  - Pointer to return the parameters

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns the parameters of a memory alloc node hNode in params_out. The poolProps and
accessDescs returned in params_out, are owned by the node. This memory remains valid until
the node is destroyed. The returned parameters must not be modified.











See also:

cudaGraphAddMemAllocNode, cudaGraphMemFreeNodeGetParams


CUDA Runtime API vRelease Version  |  405


Modules