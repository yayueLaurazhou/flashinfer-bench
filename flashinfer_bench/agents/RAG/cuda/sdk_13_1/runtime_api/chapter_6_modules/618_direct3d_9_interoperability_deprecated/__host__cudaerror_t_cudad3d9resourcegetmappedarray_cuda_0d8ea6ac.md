# __host__cudaError_t cudaD3D9ResourceGetMappedArray (cudaArray **ppArray, IDirect3DResource9 *pResource, unsigned int face, unsigned int level)

Get an array through which to access a subresource of a Direct3D resource which has been mapped for
access by CUDA.

##### Parameters

**ppArray**

  - Returned array corresponding to subresource
**pResource**

  - Mapped resource to access
**face**

  - Face of resource to access
**level**

  - Level of resource to access


CUDA Runtime API vRelease Version  |  251


Modules

##### Returns

cudaSuccess, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pArray an array through which the subresource of the mapped Direct3D resource
pResource, which corresponds to face and level may be accessed. The value set in pArray
may change every time that pResource is mapped.

If pResource is not registered then cudaErrorInvalidResourceHandle is returned. If
pResource was not registered with usage flags cudaD3D9RegisterFlagsArray, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped, then cudaErrorUnknown
is returned.

For usage requirements of face and level parameters, see cudaD3D9ResourceGetMappedPointer().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsSubResourceGetMappedArray