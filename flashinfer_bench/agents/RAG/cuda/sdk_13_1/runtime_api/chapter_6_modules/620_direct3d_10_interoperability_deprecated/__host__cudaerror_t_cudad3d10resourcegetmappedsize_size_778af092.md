# __host__cudaError_t cudaD3D10ResourceGetMappedSize (size_t *pSize, ID3D10Resource *pResource, unsigned int subResource)

Gets the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.

##### Parameters

**pSize**

  - Returned size of subresource
**pResource**

  - Mapped resource to access
**subResource**

  - Subresource of pResource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pSize the size of the subresource of the mapped Direct3D resource pResource which
corresponds to subResource. The value set in pSize may change every time that pResource is
mapped.

If pResource has not been registered for use with CUDA then cudaErrorInvalidHandle is
returned. If pResource was not registered with usage flags cudaD3D10RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped for access by CUDA then
cudaErrorUnknown is returned.

For usage requirements of the subResource parameter see
cudaD3D10ResourceGetMappedPointer().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  271


Modules


See also:

cudaGraphicsResourceGetMappedPointer