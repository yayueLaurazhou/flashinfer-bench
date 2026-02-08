# __host__cudaError_t cudaStreamGetFlags (cudaStream_t hStream, unsigned int *flags)

Query the flags of a stream.

##### Parameters

**hStream**

  - Handle to the stream to be queried


CUDA Runtime API vRelease Version  |  63


Modules


**flags**

  - Pointer to an unsigned integer in which the stream's flags are returned

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Query the flags of a stream. The flags are returned in flags. See cudaStreamCreateWithFlags for a
list of valid flags.











See also:

cudaStreamCreateWithPriority, cudaStreamCreateWithFlags, cudaStreamGetPriority,
cudaStreamGetDevice, cuStreamGetFlags