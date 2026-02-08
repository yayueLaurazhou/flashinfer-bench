# __host____device__cudaError_t cudaStreamWaitEvent (cudaStream_t stream, cudaEvent_t event, unsigned int flags)

Make a compute stream wait on an event.

##### Parameters

**stream**

  - Stream to wait
**event**

  - Event to wait on
**flags**

  - Parameters for the operation(See above)

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidResourceHandle


CUDA Runtime API vRelease Version  |  70


Modules

##### Description

Makes all future work submitted to stream wait for all work captured in event. See
cudaEventRecord() for details on what is captured by an event. The synchronization will be performed
efficiently on the device when applicable. event may be from a different device than stream.

flags include:

cudaEventWaitDefault: Default event creation flag.

##### **‣**

cudaEventWaitExternal: Event is captured in the graph as an external event node when performing

##### **‣**

stream capture.











See also:

cudaStreamCreate, cudaStreamCreateWithFlags, cudaStreamQuery, cudaStreamSynchronize,
cudaStreamAddCallback, cudaStreamDestroy, cuStreamWaitEvent