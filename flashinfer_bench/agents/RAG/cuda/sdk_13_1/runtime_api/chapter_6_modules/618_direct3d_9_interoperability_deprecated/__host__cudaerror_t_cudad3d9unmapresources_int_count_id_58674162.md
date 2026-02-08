# __host__cudaError_t cudaD3D9UnmapResources (int count, IDirect3DResource9 **ppResources)

Unmap Direct3D resources for access by CUDA.

##### Parameters

**count**

  - Number of resources to unmap for CUDA
**ppResources**

  - Resources to unmap for CUDA

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Unmaps the count Direct3D resources in ppResources.

This function provides the synchronization guarantee that any CUDA kernels issued
before cudaD3D9UnmapResources() will complete before any Direct3D calls issued after
cudaD3D9UnmapResources() begin.

If any of ppResources have not been registered for use with CUDA or if ppResources contains
any duplicate entries, then cudaErrorInvalidResourceHandle is returned. If any of ppResources are
not presently mapped for access by CUDA then cudaErrorUnknown is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsUnmapResources


CUDA Runtime API vRelease Version  |  258


Modules