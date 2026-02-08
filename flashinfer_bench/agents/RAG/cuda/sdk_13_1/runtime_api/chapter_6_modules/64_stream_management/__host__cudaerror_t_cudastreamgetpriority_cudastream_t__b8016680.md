# __host__cudaError_t cudaStreamGetPriority (cudaStream_t hStream, int *priority)

Query the priority of a stream.

##### Parameters

**hStream**

  - Handle to the stream to be queried
**priority**

  - Pointer to a signed integer in which the stream's priority is returned

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle

##### Description

Query the priority of a stream. The priority is returned in in priority. Note that if
the stream was created with a priority outside the meaningful numerical range returned
by cudaDeviceGetStreamPriorityRange, this function returns the clamped priority. See
cudaStreamCreateWithPriority for details about priority clamping.


CUDA Runtime API vRelease Version  |  65


Modules


See also:

cudaStreamCreateWithPriority, cudaDeviceGetStreamPriorityRange, cudaStreamGetFlags,
cudaStreamGetDevice, cudaStreamGetDevResource, cuStreamGetPriority