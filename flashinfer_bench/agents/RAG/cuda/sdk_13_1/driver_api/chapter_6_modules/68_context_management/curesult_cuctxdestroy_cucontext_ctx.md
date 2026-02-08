# CUresult cuCtxDestroy (CUcontext ctx)

Destroy a CUDA context.

###### Parameters

**ctx**

  - Context to destroy

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

###### Description

Destroys the CUDA context specified by ctx. The context ctx will be destroyed regardless of how
many threads it is current to. It is the responsibility of the calling function to ensure that no API call
issues using ctx while cuCtxDestroy() is executing.

Destroys and cleans up all resources associated with the context. It is the caller's responsibility to
ensure that the context or its resources are not accessed or passed in subsequent API calls and doing
so will result in undefined behavior. These resources include CUDA types CUmodule, CUfunction,
CUstream, CUevent, CUarray, CUmipmappedArray, CUtexObject, CUsurfObject, CUtexref,
CUsurfref, CUgraphicsResource, CUlinkState, CUexternalMemory and CUexternalSemaphore.
These resources also include memory allocations by cuMemAlloc(), cuMemAllocHost(),
cuMemAllocManaged() and cuMemAllocPitch().

If ctx is current to the calling thread then ctx will also be popped from the current thread's context
stack (as though cuCtxPopCurrent() were called). If ctx is current to other threads, then ctx will
remain current to those threads, and attempting to access ctx from those threads will result in the error
CUDA_ERROR_CONTEXT_IS_DESTROYED.





Note:


Note that this function may also return error codes from previous, asynchronous launches.


CUDA Driver API TRM-06703-001 _vRelease Version  |  123


Modules


See also:

cuCtxCreate, cuCtxGetApiVersion, cuCtxGetCacheConfig, cuCtxGetDevice, cuCtxGetFlags,
cuCtxGetLimit, cuCtxPopCurrent, cuCtxPushCurrent, cuCtxSetCacheConfig, cuCtxSetLimit,
cuCtxSynchronize