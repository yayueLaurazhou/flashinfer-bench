# __host__cudaError_t cudaD3D9ResourceGetMappedSize (size_t *pSize, IDirect3DResource9 *pResource, unsigned int face, unsigned int level)

Get the size of a subresource of a Direct3D resource which has been mapped for access by CUDA.

##### Parameters

**pSize**

  - Returned size of subresource
**pResource**

  - Mapped resource to access
**face**

  - Face of resource to access
**level**

  - Level of resource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pSize the size of the subresource of the mapped Direct3D resource pResource, which
corresponds to face and level. The value set in pSize may change every time that pResource is
mapped.

If pResource has not been registered for use with CUDA then cudaErrorInvalidResourceHandle
is returned. If pResource was not registered with usage flags cudaD3D9RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped for access by CUDA then
cudaErrorUnknown is returned.

For usage requirements of face and level parameters, see cudaD3D9ResourceGetMappedPointer().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Runtime API vRelease Version  |  255


Modules


cudaGraphicsResourceGetMappedPointer