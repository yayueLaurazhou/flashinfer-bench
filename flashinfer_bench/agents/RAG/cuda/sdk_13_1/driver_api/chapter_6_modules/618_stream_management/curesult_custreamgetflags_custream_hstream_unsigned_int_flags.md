# CUresult cuStreamGetFlags (CUstream hStream, unsigned int *flags)

Query the flags of a given stream.

###### Parameters

**hStream**

  - Handle to the stream to be queried
**flags**

  - Pointer to an unsigned integer in which the stream's flags are returned The value returned in
flags is a logical 'OR' of all flags that were used while creating this stream. See cuStreamCreate
for the list of valid flags

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_HANDLE, CUDA_ERROR_OUT_OF_MEMORY

###### Description

Query the flags of a stream created using cuStreamCreate, cuStreamCreateWithPriority or
cuGreenCtxStreamCreate and return the flags in flags.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamDestroy, cuStreamCreate, cuGreenCtxStreamCreate, cuStreamGetPriority,
cudaStreamGetFlags, cuStreamGetDevice


CUDA Driver API TRM-06703-001 _vRelease Version  |  346


Modules