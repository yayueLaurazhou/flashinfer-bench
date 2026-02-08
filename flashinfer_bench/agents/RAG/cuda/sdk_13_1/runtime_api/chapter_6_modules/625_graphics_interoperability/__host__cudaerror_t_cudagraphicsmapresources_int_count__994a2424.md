# __host__cudaError_t cudaGraphicsMapResources (int count, cudaGraphicsResource_t *resources, cudaStream_t stream)

Map graphics resources for access by CUDA.

##### Parameters

**count**

  - Number of resources to map
**resources**

  - Resources to map for CUDA
**stream**

  - Stream for synchronization

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Maps the count graphics resources in resources for access by CUDA.

The resources in resources may be accessed by CUDA until they are unmapped. The graphics API
from which resources were registered should not access any resources while they are mapped by
CUDA. If an application does so, the results are undefined.

This function provides the synchronization guarantee that any graphics calls issued before
cudaGraphicsMapResources() will complete before any subsequent CUDA work issued in stream
begins.

If resources contains any duplicate entries then cudaErrorInvalidResourceHandle is returned. If any
of resources are presently mapped for access by CUDA then cudaErrorUnknown is returned.











See also:


CUDA Runtime API vRelease Version  |  296


Modules


cudaGraphicsResourceGetMappedPointer, cudaGraphicsSubResourceGetMappedArray,
cudaGraphicsUnmapResources, cuGraphicsMapResources