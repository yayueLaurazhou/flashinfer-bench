# __host__cudaError_t cudaD3D10ResourceGetMappedPitch (size_t *pPitch, size_t *pPitchSlice, ID3D10Resource *pResource, unsigned int subResource)

Gets the pitch of a subresource of a Direct3D resource which has been mapped for access by CUDA.

##### Parameters

**pPitch**

  - Returned pitch of subresource
**pPitchSlice**

  - Returned Z-slice pitch of subresource
**pResource**

  - Mapped resource to access
**subResource**

  - Subresource of pResource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pPitch and *pPitchSlice the pitch and Z-slice pitch of the subresource of the
mapped Direct3D resource pResource, which corresponds to subResource. The values set in
pPitch and pPitchSlice may change every time that pResource is mapped.

The pitch and Z-slice pitch values may be used to compute the location of a sample on a surface as
follows.

For a 2D surface, the byte offset of the sample at position x, y from the base pointer of the surface is:

y * pitch + (bytes per pixel) * x

For a 3D surface, the byte offset of the sample at position x, y, z from the base pointer of the surface is:

z* slicePitch + y * pitch + (bytes per pixel) * x

Both parameters pPitch and pPitchSlice are optional and may be set to NULL.

If pResource is not of type ID3D10Texture1D, ID3D10Texture2D, or ID3D10Texture3D, or if
pResource has not been registered for use with CUDA, then cudaErrorInvalidResourceHandle is
returned. If pResource was not registered with usage flags cudaD3D10RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped for access by CUDA then
cudaErrorUnknown is returned.

For usage requirements of the subResource parameter see
cudaD3D10ResourceGetMappedPointer().


CUDA Runtime API vRelease Version  |  269


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsSubResourceGetMappedArray