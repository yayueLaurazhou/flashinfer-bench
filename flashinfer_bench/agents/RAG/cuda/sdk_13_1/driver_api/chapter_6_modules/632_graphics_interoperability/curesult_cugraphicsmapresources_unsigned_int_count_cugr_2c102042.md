# CUresult cuGraphicsMapResources (unsigned int count, CUgraphicsResource *resources, CUstream hStream)

Map graphics resources for access by CUDA.

###### Parameters

**count**

  - Number of resources to map
**resources**

  - Resources to map for CUDA usage
**hStream**

  - Stream with which to synchronize

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_HANDLE,
CUDA_ERROR_ALREADY_MAPPED, CUDA_ERROR_UNKNOWN

###### Description

Maps the count graphics resources in resources for access by CUDA.

The resources in resources may be accessed by CUDA until they are unmapped. The graphics API
from which resources were registered should not access any resources while they are mapped by
CUDA. If an application does so, the results are undefined.

This function provides the synchronization guarantee that any graphics calls issued before
cuGraphicsMapResources() will complete before any subsequent CUDA work issued in stream
begins.


CUDA Driver API TRM-06703-001 _vRelease Version  |  556


Modules



If resources includes any duplicate entries then CUDA_ERROR_INVALID_HANDLE
is returned. If any of resources are presently mapped for access by CUDA then
CUDA_ERROR_ALREADY_MAPPED is returned.





See also:

cuGraphicsResourceGetMappedPointer, cuGraphicsSubResourceGetMappedArray,
cuGraphicsUnmapResources, cudaGraphicsMapResources