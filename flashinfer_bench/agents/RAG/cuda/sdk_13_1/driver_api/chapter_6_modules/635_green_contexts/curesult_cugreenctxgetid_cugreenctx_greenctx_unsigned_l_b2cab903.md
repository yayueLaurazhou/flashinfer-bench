# CUresult cuGreenCtxGetId (CUgreenCtx greenCtx, unsigned long long *greenCtxId)

Returns the unique Id associated with the green context supplied.

###### Parameters

**greenCtx**

  - Green context for which to obtain the Id
**greenCtxId**

  - Pointer to store the Id of the green context

###### Returns

CUDA_SUCCESS, CUDA_ERROR_CONTEXT_IS_DESTROYED,
CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in greenCtxId the unique Id which is associated with a given green context. The Id is
unique for the life of the program for this instance of CUDA. If green context is supplied as NULL and
the current context is set to a green context, the Id of the current green context is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuGreenCtxCreate, cuGreenCtxDestroy, cuCtxGetId


CUDA Driver API TRM-06703-001 _vRelease Version  |  585


Modules