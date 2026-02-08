# __host__cudaError_t cudaD3D10ResourceSetMapFlags (ID3D10Resource *pResource, unsigned int flags)

Set usage flags for mapping a Direct3D resource.

##### Parameters

**pResource**

  - Registered resource to set flags for
**flags**

  - Parameters for resource mapping

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown,

##### Description

Deprecated This function is deprecated as of CUDA 3.0.

Set usage flags for mapping the Direct3D resource pResource.

Changes to flags will take effect the next time pResource is mapped. The flags argument may be
any of the following:

cudaD3D10MapFlagsNone: Specifies no hints about how this resource will be used. It is therefore

##### **‣**

assumed that this resource will be read from and written to by CUDA kernels. This is the default
value.
cudaD3D10MapFlagsReadOnly: Specifies that CUDA kernels which access this resource will not

##### **‣**

write to this resource.
cudaD3D10MapFlagsWriteDiscard: Specifies that CUDA kernels which access this resource will

##### **‣**

not read from this resource and will write over the entire contents of the resource, so none of the
data previously stored in the resource will be preserved.

If pResource has not been registered for use with CUDA then cudaErrorInvalidHandle is returned. If
pResource is presently mapped for access by CUDA then cudaErrorUnknown is returned.


Note:


CUDA Runtime API vRelease Version  |  273


Modules


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaGraphicsResourceSetMapFlags