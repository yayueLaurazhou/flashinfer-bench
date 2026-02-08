# CUresult cuMemHostGetFlags (unsigned int *pFlags, void *p)

Passes back flags that were used for a pinned allocation.

###### Parameters

**pFlags**

  - Returned flags word
**p**

  - Host pointer

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Passes back the flags pFlags that were specified when allocating the pinned host buffer p allocated
by cuMemHostAlloc.

cuMemHostGetFlags() will fail if the pointer does not reside in an allocation performed by
cuMemAllocHost() or cuMemHostAlloc().


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuMemAllocHost, cuMemHostAlloc, cudaHostGetFlags