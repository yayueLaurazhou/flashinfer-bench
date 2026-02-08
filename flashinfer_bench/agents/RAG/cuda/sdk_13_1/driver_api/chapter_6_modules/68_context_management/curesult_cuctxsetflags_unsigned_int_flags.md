# CUresult cuCtxSetFlags (unsigned int flags)

Sets the flags for the current context.

###### Parameters

**flags**

  - Flags to set on the current context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,

###### Description

Sets the flags for the current context overwriting previously set ones. See cuDevicePrimaryCtxSetFlags
for flag values.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetCurrent, cuCtxGetDevice,
cuCtxGetLimit, cuCtxGetSharedMemConfig, cuCtxGetStreamPriorityRange, cuCtxGetFlags,
cudaGetDeviceFlags, cuDevicePrimaryCtxSetFlags,