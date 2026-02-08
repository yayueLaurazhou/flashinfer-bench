# __host__cudaError_t cudaD3D10ResourceGetMappedArray (cudaArray **ppArray, ID3D10Resource *pResource, unsigned int subResource)

Gets an array through which to access a subresource of a Direct3D resource which has been mapped
for access by CUDA.

##### Parameters

**ppArray**

  - Returned array corresponding to subresource
**pResource**

  - Mapped resource to access
**subResource**

  - Subresource of pResource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *ppArray an array through which the subresource of the mapped Direct3D resource
pResource which corresponds to subResource may be accessed. The value set in ppArray may
change every time that pResource is mapped.

If pResource is not registered, then cudaErrorInvalidResourceHandle is returned. If
pResource was not registered with usage flags cudaD3D10RegisterFlagsArray, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped then cudaErrorUnknown
is returned.

For usage requirements of the subResource parameter, see
cudaD3D10ResourceGetMappedPointer().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsSubResourceGetMappedArray


CUDA Runtime API vRelease Version  |  268


Modules