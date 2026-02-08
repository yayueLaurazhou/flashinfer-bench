# __host__cudaError_t cudaD3D9ResourceGetMappedPointer (void **pPointer, IDirect3DResource9 *pResource, unsigned int face, unsigned int level)

Get a pointer through which to access a subresource of a Direct3D resource which has been mapped for
access by CUDA.

##### Parameters

**pPointer**

  - Returned pointer corresponding to subresource
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

Returns in *pPointer the base pointer of the subresource of the mapped Direct3D resource
pResource, which corresponds to face and level. The value set in pPointer may change every
time that pResource is mapped.

If pResource is not registered, then cudaErrorInvalidResourceHandle is returned. If
pResource was not registered with usage flags cudaD3D9RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped, then cudaErrorUnknown
is returned.

If pResource is of type IDirect3DCubeTexture9, then face must one of the values enumerated
by type D3DCUBEMAP_FACES. For all other types, face must be 0. If face is invalid, then
cudaErrorInvalidValue is returned.

If pResource is of type IDirect3DBaseTexture9, then level must correspond to a valid mipmap
level. Only mipmap level 0 is supported for now. For all other types level must be 0. If level is
invalid, then cudaErrorInvalidValue is returned.


Note:


CUDA Runtime API vRelease Version  |  254


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsResourceGetMappedPointer