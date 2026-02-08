# CUresult cuDestroyExternalMemory (CUexternalMemory extMem)

Destroys an external memory object.

###### Parameters

**extMem**

  - External memory object to be destroyed

###### Returns

CUDA_SUCCESS, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_HANDLE

###### Description

Destroys the specified external memory object. Any existing buffers and CUDA mipmapped arrays
mapped onto this object must no longer be used and must be explicitly freed using cuMemFree and
cuMipmappedArrayDestroy respectively.


CUDA Driver API TRM-06703-001 _vRelease Version  |  361


Modules


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuImportExternalMemory, cuExternalMemoryGetMappedBuffer,
cuExternalMemoryGetMappedMipmappedArray