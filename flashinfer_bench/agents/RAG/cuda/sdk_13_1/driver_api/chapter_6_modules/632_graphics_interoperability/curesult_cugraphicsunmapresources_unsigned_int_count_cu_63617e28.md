# CUresult cuGraphicsUnmapResources (unsigned int count, CUgraphicsResource *resources, CUstream hStream)

Unmap graphics resources.

###### Parameters

**count**

  - Number of resources to unmap
**resources**

  - Resources to unmap
**hStream**

  - Stream with which to synchronize

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_NOT_MAPPED, CUDA_ERROR_UNKNOWN

###### Description

Unmaps the count graphics resources in resources.

Once unmapped, the resources in resources may not be accessed by CUDA until they are mapped
again.

This function provides the synchronization guarantee that any CUDA work issued in stream before
cuGraphicsUnmapResources() will complete before any subsequently issued graphics work begins.

If resources includes any duplicate entries then CUDA_ERROR_INVALID_HANDLE
is returned. If any of resources are not presently mapped for access by CUDA then
CUDA_ERROR_NOT_MAPPED is returned.





See also:

cuGraphicsMapResources, cudaGraphicsUnmapResources


CUDA Driver API TRM-06703-001 _vRelease Version  |  561


Modules