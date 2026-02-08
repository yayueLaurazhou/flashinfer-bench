# __host__cudaError_t cudaGraphKernelNodeGetParams (cudaGraphNode_t node, cudaKernelNodeParams *pNodeParams)

Returns a kernel node's parameters.

##### Parameters

**node**

  - Node to get the parameters for
**pNodeParams**

  - Pointer to return the parameters


CUDA Runtime API vRelease Version  |  396


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction

##### Description

Returns the parameters of kernel node node in pNodeParams. The kernelParams or extra
array returned in pNodeParams, as well as the argument values it points to, are owned by the node.
This memory remains valid until the node is destroyed or its parameters are modified, and should not
be modified directly. Use cudaGraphKernelNodeSetParams to update the parameters of this node.

The params will contain either kernelParams or extra, according to which of these was most
recently set on the node.











See also:

cudaLaunchKernel, cudaGraphAddKernelNode, cudaGraphKernelNodeSetParams