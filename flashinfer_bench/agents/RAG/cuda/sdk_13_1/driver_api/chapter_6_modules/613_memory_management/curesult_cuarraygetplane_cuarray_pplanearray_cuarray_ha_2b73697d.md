# CUresult cuArrayGetPlane (CUarray *pPlaneArray, CUarray hArray, unsigned int planeIdx)

Gets a CUDA array plane from a CUDA array.

###### Parameters

**pPlaneArray**

  - Returned CUDA array referenced by the planeIdx
**hArray**

  - Multiplanar CUDA array


CUDA Driver API TRM-06703-001 _vRelease Version  |  185


Modules


**planeIdx**

  - Plane index

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE

###### Description

Returns in pPlaneArray a CUDA array that represents a single format plane of the CUDA array
hArray.

If planeIdx is greater than the maximum number of planes in this array or if the array does not have
a multi-planar format e.g: CU_AD_FORMAT_NV12, then CUDA_ERROR_INVALID_VALUE is
returned.

Note that if the hArray has format CU_AD_FORMAT_NV12, then passing in 0 for
planeIdx returns a CUDA array of the same size as hArray but with one channel and
CU_AD_FORMAT_UNSIGNED_INT8 as its format. If 1 is passed for planeIdx, then
the returned CUDA array has half the height and width of hArray with two channels and
CU_AD_FORMAT_UNSIGNED_INT8 as its format.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuArrayCreate, cudaArrayGetPlane