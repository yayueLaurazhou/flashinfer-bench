# __host__cudaError_t cudaD3D9ResourceSetMapFlags (IDirect3DResource9 *pResource, unsigned int flags)

Set usage flags for mapping a Direct3D resource.

##### Parameters

**pResource**

  - Registered resource to set flags for
**flags**

  - Parameters for resource mapping

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Set flags for mapping the Direct3D resource pResource.

Changes to flags will take effect the next time pResource is mapped. The flags argument may be
any of the following:

cudaD3D9MapFlagsNone: Specifies no hints about how this resource will be used. It is therefore

##### **‣**

assumed that this resource will be read from and written to by CUDA kernels. This is the default
value.
cudaD3D9MapFlagsReadOnly: Specifies that CUDA kernels which access this resource will not

##### **‣**

write to this resource.
cudaD3D9MapFlagsWriteDiscard: Specifies that CUDA kernels which access this resource will

##### **‣**

not read from this resource and will write over the entire contents of the resource, so none of the
data previously stored in the resource will be preserved.

If pResource has not been registered for use with CUDA, then cudaErrorInvalidResourceHandle
is returned. If pResource is presently mapped for access by CUDA, then cudaErrorUnknown is
returned.


CUDA Runtime API vRelease Version  |  257


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaInteropResourceSetMapFlags