# CUresult cuCtxGetApiVersion (CUcontext ctx, unsigned int *version)

Gets the context's API version.

###### Parameters

**ctx**

  - Context to check
**version**

  - Pointer to version

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_UNKNOWN

###### Description

Returns a version number in version corresponding to the capabilities of the context (e.g. 3010 or
3020), which library developers can use to direct callers to a specific API version. If ctx is NULL,
returns the API version used to create the currently bound context.

Note that new API versions are only introduced when context capabilities are changed that break
binary compatibility, so the API version and driver version may be different. For example, it is valid
for the API version to be 3020 while the driver version is 4020.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuCtxCreate, cuCtxDestroy, cuCtxGetDevice, cuCtxGetFlags, cuCtxGetLimit, cuCtxPopCurrent,
cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit, cuCtxSynchronize


CUDA Driver API TRM-06703-001 _vRelease Version  |  124


Modules