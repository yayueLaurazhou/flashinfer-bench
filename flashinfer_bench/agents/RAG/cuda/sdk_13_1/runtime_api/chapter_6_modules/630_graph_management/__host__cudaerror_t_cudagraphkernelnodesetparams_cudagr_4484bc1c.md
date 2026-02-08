# __host__cudaError_t cudaGraphKernelNodeSetParams (cudaGraphNode_t node, const cudaKernelNodeParams *pNodeParams)

Sets a kernel node's parameters.

##### Parameters

**node**

  - Node to set the parameters for
**pNodeParams**

  - Parameters to copy

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation

##### Description

Sets the parameters of kernel node node to pNodeParams.















See also:

cudaGraphNodeSetParams, cudaLaunchKernel, cudaGraphAddKernelNode,
cudaGraphKernelNodeGetParams


CUDA Runtime API vRelease Version  |  402


Modules