# CUresult cuCtxGetFlags (unsigned int *flags)

Returns the flags for the current context.

###### Parameters

**flags**

  - Pointer to store flags of current context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

###### Description

Returns in *flags the flags of the current context. See cuCtxCreate for flag values.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  128


Modules


cuCtxCreate, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetCurrent, cuCtxGetDevice,
cuCtxGetLimit, cuCtxGetSharedMemConfig, cuCtxGetStreamPriorityRange, cuCtxSetFlags,
cudaGetDeviceFlags