# __host__cudaError_t cudaD3D9ResourceGetMappedPitch (size_t *pPitch, size_t *pPitchSlice, IDirect3DResource9 *pResource, unsigned int face, unsigned int level)

Get the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.

##### Parameters

**pPitch**

  - Returned pitch of subresource
**pPitchSlice**

  - Returned Z-slice pitch of subresource
**pResource**

  - Mapped resource to access
**face**

  - Face of resource to access
**level**

  - Level of resource to access


CUDA Runtime API vRelease Version  |  252


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pPitch and *pPitchSlice the pitch and Z-slice pitch of the subresource of the
mapped Direct3D resource pResource, which corresponds to face and level. The values set in
pPitch and pPitchSlice may change every time that pResource is mapped.

The pitch and Z-slice pitch values may be used to compute the location of a sample on a surface as
follows.

For a 2D surface, the byte offset of the sample at position x, y from the base pointer of the surface is:

y * pitch + (bytes per pixel) * x

For a 3D surface, the byte offset of the sample at position x, y, z from the base pointer of the surface is:

z* slicePitch + y * pitch + (bytes per pixel) * x

Both parameters pPitch and pPitchSlice are optional and may be set to NULL.

If pResource is not of type IDirect3DBaseTexture9 or one of its sub-types or if pResource
has not been registered for use with CUDA, then cudaErrorInvalidResourceHandle is returned.
If pResource was not registered with usage flags cudaD3D9RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped for access by CUDA then
cudaErrorUnknown is returned.

For usage requirements of face and level parameters, see cudaD3D9ResourceGetMappedPointer().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsResourceGetMappedPointer


CUDA Runtime API vRelease Version  |  253


Modules