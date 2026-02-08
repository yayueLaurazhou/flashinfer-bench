# CUresult cuGraphKernelNodeSetParams (CUgraphNode hNode, const CUDA_KERNEL_NODE_PARAMS *nodeParams)

Sets a kernel node's parameters.

###### Parameters

**hNode**

  - Node to set the parameters for
**nodeParams**

  - Parameters to copy


CUDA Driver API TRM-06703-001 _vRelease Version  |  477


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_OUT_OF_MEMORY

###### Description

Sets the parameters of kernel node hNode to nodeParams.





See also:

cuGraphNodeSetParams, cuLaunchKernel, cuGraphAddKernelNode, cuGraphKernelNodeGetParams