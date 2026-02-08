# __host__cudaError_t cudaD3D10ResourceGetMappedPointer (void **pPointer, ID3D10Resource *pResource, unsigned int subResource)

Gets a pointer through which to access a subresource of a Direct3D resource which has been mapped
for access by CUDA.

##### Parameters

**pPointer**

  - Returned pointer corresponding to subresource
**pResource**

  - Mapped resource to access
**subResource**

  - Subresource of pResource to access

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Returns in *pPointer the base pointer of the subresource of the mapped Direct3D resource
pResource which corresponds to subResource. The value set in pPointer may change every
time that pResource is mapped.

If pResource is not registered, then cudaErrorInvalidResourceHandle is returned. If
pResource was not registered with usage flags cudaD3D9RegisterFlagsNone, then
cudaErrorInvalidResourceHandle is returned. If pResource is not mapped then cudaErrorUnknown
is returned.

If pResource is of type ID3D10Buffer then subResource must be 0. If pResource is of
any other type, then the value of subResource must come from the subresource calculation in
D3D10CalcSubResource().


CUDA Runtime API vRelease Version  |  270


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsResourceGetMappedPointer