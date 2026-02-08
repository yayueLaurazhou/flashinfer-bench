# __host__cudaError_t cudaGraphicsResourceSetMapFlags (cudaGraphicsResource_t resource, unsigned int flags)

Set usage flags for mapping a graphics resource.

##### Parameters

**resource**

  - Registered resource to set flags for
**flags**

  - Parameters for resource mapping

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle, cudaErrorUnknown,

##### Description

Set flags for mapping the graphics resource resource.

Changes to flags will take effect the next time resource is mapped. The flags argument may be
any of the following:

cudaGraphicsMapFlagsNone: Specifies no hints about how will be used. It is therefore

##### ‣ resource

assumed that CUDA may read from or write to resource.
cudaGraphicsMapFlagsReadOnly: Specifies that CUDA will not write to .

##### ‣ resource

cudaGraphicsMapFlagsWriteDiscard: Specifies CUDA will not read from and will

##### ‣ resource

write over the entire contents of resource, so none of the data previously stored in resource
will be preserved.

If resource is presently mapped for access by CUDA then cudaErrorUnknown is returned. If
flags is not one of the above values then cudaErrorInvalidValue is returned.



See also:

cudaGraphicsMapResources, cuGraphicsResourceSetMapFlags


CUDA Runtime API vRelease Version  |  299


Modules