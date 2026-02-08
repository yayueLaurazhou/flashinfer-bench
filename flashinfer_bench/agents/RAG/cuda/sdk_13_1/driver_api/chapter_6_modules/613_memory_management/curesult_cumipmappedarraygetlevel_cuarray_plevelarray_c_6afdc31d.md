# CUresult cuMipmappedArrayGetLevel (CUarray *pLevelArray, CUmipmappedArray hMipmappedArray, unsigned int level)

Gets a mipmap level of a CUDA mipmapped array.

###### Parameters

**pLevelArray**

  - Returned mipmap level CUDA array
**hMipmappedArray**

  - CUDA mipmapped array
**level**

  - Mipmap level


CUDA Driver API TRM-06703-001 _vRelease Version  |  268


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Returns in *pLevelArray a CUDA array that represents a single mipmap level of the CUDA
mipmapped array hMipmappedArray.

If level is greater than the maximum number of levels in this mipmapped array,
CUDA_ERROR_INVALID_VALUE is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMipmappedArrayCreate, cuMipmappedArrayDestroy, cuArrayCreate,
cudaGetMipmappedArrayLevel