# __host__cudaError_t cudaHostGetFlags (unsigned int *pFlags, void *pHost)

Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.

##### Parameters

**pFlags**

  - Returned flags word
**pHost**

  - Host pointer

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  130


Modules

##### Description

cudaHostGetFlags() will fail if the input pointer does not reside in an address range allocated by
cudaHostAlloc().



See also:

cudaHostAlloc, cuMemHostGetFlags