# CUresult cuGraphicsResourceSetMapFlags (CUgraphicsResource resource, unsigned int flags)

Set usage flags for mapping a graphics resource.

###### Parameters

**resource**

  - Registered resource to set flags for
**flags**

  - Parameters for resource mapping

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_ALREADY_MAPPED

###### Description

Set flags for mapping the graphics resource resource.

Changes to flags will take effect the next time resource is mapped. The flags argument may be
any of the following:

CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: Specifies no hints about how this resource

###### **‣**

will be used. It is therefore assumed that this resource will be read from and written to by CUDA
kernels. This is the default value.
CU_GRAPHICS_MAP_RESOURCE_FLAGS_READONLY: Specifies that CUDA kernels which

###### **‣**

access this resource will not write to this resource.
CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITEDISCARD: Specifies that CUDA kernels

###### **‣**

which access this resource will not read from this resource and will write over the entire contents of
the resource, so none of the data previously stored in the resource will be preserved.

If resource is presently mapped for access by CUDA then CUDA_ERROR_ALREADY_MAPPED
is returned. If flags is not one of the above values then CUDA_ERROR_INVALID_VALUE is
returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  559


Modules


cuGraphicsMapResources, cudaGraphicsResourceSetMapFlags