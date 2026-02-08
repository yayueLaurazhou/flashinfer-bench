# CUresult cuGetErrorString (CUresult error, const char **pStr)

Gets the string description of an error code.

###### Parameters

**error**

  - Error code to convert to string
**pStr**

  - Address of the string pointer.


CUDA Driver API TRM-06703-001 _vRelease Version  |  98


Modules

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Sets *pStr to the address of a NULL-terminated string description of the error code error. If the
error code is not recognized, CUDA_ERROR_INVALID_VALUE will be returned and *pStr will be
set to the NULL address.


See also:

CUresult, cudaGetErrorString