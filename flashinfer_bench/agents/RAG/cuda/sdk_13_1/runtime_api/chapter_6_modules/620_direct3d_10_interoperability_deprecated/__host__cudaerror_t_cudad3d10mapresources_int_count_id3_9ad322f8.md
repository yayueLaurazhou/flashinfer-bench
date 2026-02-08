# __host__cudaError_t cudaD3D10MapResources (int count, ID3D10Resource **ppResources)

Maps Direct3D Resources for access by CUDA.

##### Parameters

**count**

  - Number of resources to map for CUDA
**ppResources**

  - Resources to map for CUDA

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Maps the count Direct3D resources in ppResources for access by CUDA.

The resources in ppResources may be accessed in CUDA kernels until they are unmapped.
Direct3D should not access any resources while they are mapped by CUDA. If an application does so,
the results are undefined.

This function provides the synchronization guarantee that any Direct3D calls issued
before cudaD3D10MapResources() will complete before any CUDA kernels issued after
cudaD3D10MapResources() begin.


CUDA Runtime API vRelease Version  |  265


Modules


If any of ppResources have not been registered for use with CUDA or if ppResources contains
any duplicate entries then cudaErrorInvalidResourceHandle is returned. If any of ppResources are
presently mapped for access by CUDA then cudaErrorUnknown is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsMapResources