# __host__cudaError_t cudaGraphicsUnmapResources (int count, cudaGraphicsResource_t *resources, cudaStream_t stream)

Unmap graphics resources.

##### Parameters

**count**

  - Number of resources to unmap
**resources**

  - Resources to unmap
**stream**

  - Stream for synchronization

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Unmaps the count graphics resources in resources.

Once unmapped, the resources in resources may not be accessed by CUDA until they are mapped
again.

This function provides the synchronization guarantee that any CUDA work issued in stream before
cudaGraphicsUnmapResources() will complete before any subsequently issued graphics work begins.

If resources contains any duplicate entries then cudaErrorInvalidResourceHandle is returned. If any
of resources are not presently mapped for access by CUDA then cudaErrorUnknown is returned.











See also:

cudaGraphicsMapResources, cuGraphicsUnmapResources


CUDA Runtime API vRelease Version  |  301


Modules