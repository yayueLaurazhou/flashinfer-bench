# CUresult cuGraphKernelNodeGetParams (CUgraphNode hNode, CUDA_KERNEL_NODE_PARAMS *nodeParams)

Returns a kernel node's parameters.

###### Parameters

**hNode**

  - Node to get the parameters for
**nodeParams**

  - Pointer to return the parameters

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE

###### Description

Returns the parameters of kernel node hNode in nodeParams. The kernelParams or extra
array returned in nodeParams, as well as the argument values it points to, are owned by the node.
This memory remains valid until the node is destroyed or its parameters are modified, and should not
be modified directly. Use cuGraphKernelNodeSetParams to update the parameters of this node.

The params will contain either kernelParams or extra, according to which of these was most
recently set on the node.





CUDA Driver API TRM-06703-001 _vRelease Version  |  476


Modules


See also:

cuLaunchKernel, cuGraphAddKernelNode, cuGraphKernelNodeSetParams