# CUresult cuStreamGetPriority (CUstream hStream, int *priority)

Query the priority of a given stream.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**priority**

  - Pointer to a signed integer in which the stream's priority is returned

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Query the priority of a stream created using cuStreamCreate, cuStreamCreateWithPriority or
cuGreenCtxStreamCreate and return the priority in priority. Note that if the stream was created
with a priority outside the numerical range returned by cuCtxGetStreamPriorityRange, this function
returns the clamped priority. See cuStreamCreateWithPriority for details about priority clamping.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreate, cuStreamCreateWithPriority, cuGreenCtxStreamCreate,
cuCtxGetStreamPriorityRange, cuStreamGetFlags, cuStreamGetDevice, cudaStreamGetPriority