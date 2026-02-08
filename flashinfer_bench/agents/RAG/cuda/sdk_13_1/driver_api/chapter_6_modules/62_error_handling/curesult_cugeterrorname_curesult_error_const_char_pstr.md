# CUresult cuGetErrorName (CUresult error, const char **pStr)

Gets the string representation of an error code enum name.

###### Parameters

**error**

  - Error code to convert to string
**pStr**

  - Address of the string pointer.

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Sets *pStr to the address of a NULL-terminated string representation of the name of the enum error
code error. If the error code is not recognized, CUDA_ERROR_INVALID_VALUE will be returned
and *pStr will be set to the NULL address.


See also:

CUresult, cudaGetErrorName