# CUresult cuMemRelease (CUmemGenericAllocationHandle handle)

Release a memory handle representing a memory allocation which was previously allocated through
cuMemCreate.

###### Parameters

**handle**
Value of handle which was returned previously by cuMemCreate.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_PERMITTED,
CUDA_ERROR_NOT_SUPPORTED

###### Description

Frees the memory that was allocated on a device through cuMemCreate.

The memory allocation will be freed when all outstanding mappings to the memory are unmapped and
when all outstanding references to the handle (including it's shareable counterparts) are also released.
The generic memory handle can be freed when there are still outstanding mappings made with this
handle. Each time a recipient process imports a shareable handle, it needs to pair it with cuMemRelease
for the handle to be freed. If handle is not a valid handle the behavior is undefined.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemCreate