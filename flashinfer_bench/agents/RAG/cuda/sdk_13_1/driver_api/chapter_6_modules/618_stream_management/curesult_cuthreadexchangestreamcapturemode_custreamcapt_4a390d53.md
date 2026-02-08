# CUresult cuThreadExchangeStreamCaptureMode (CUstreamCaptureMode *mode)

Swaps the stream capture interaction mode for a thread.

###### Parameters

**mode**

  - Pointer to mode value to swap with the current mode

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE


CUDA Driver API TRM-06703-001 _vRelease Version  |  353


Modules

###### Description

Sets the calling thread's stream capture interaction mode to the value contained in *mode, and
overwrites *mode with the previous mode for the thread. To facilitate deterministic behavior across
function or module boundaries, callers are encouraged to use this API in a push-pop fashion:


During stream capture (see cuStreamBeginCapture), some actions, such as a call to cudaMalloc,
may be unsafe. In the case of cudaMalloc, the operation is not enqueued asynchronously to a
stream, and is not observed by stream capture. Therefore, if the sequence of operations captured via
cuStreamBeginCapture depended on the allocation being replayed whenever the graph is launched, the
captured graph would be invalid.

Therefore, stream capture places restrictions on API calls that can be made within or concurrently to a
cuStreamBeginCapture-cuStreamEndCapture sequence. This behavior can be controlled via this API
and flags to cuStreamBeginCapture.

A thread's mode is one of the following:

This is the default mode. If the local thread has an

###### ‣ CU_STREAM_CAPTURE_MODE_GLOBAL:

ongoing capture sequence that was not initiated with CU_STREAM_CAPTURE_MODE_RELAXED
at cuStreamBeginCapture, or if any other thread has a concurrent capture sequence initiated
with CU_STREAM_CAPTURE_MODE_GLOBAL, this thread is prohibited from potentially unsafe
API calls.
If the local thread has an ongoing capture

###### ‣ CU_STREAM_CAPTURE_MODE_THREAD_LOCAL:

sequence not initiated with CU_STREAM_CAPTURE_MODE_RELAXED, it is prohibited from
potentially unsafe API calls. Concurrent capture sequences in other threads are ignored.
The local thread is not prohibited from potentially

###### ‣ CU_STREAM_CAPTURE_MODE_RELAXED:

unsafe API calls. Note that the thread is still prohibited from API calls which necessarily conflict
with stream capture, for example, attempting cuEventQuery on an event that was last recorded
inside a capture sequence.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuStreamBeginCapture


CUDA Driver API TRM-06703-001 _vRelease Version  |  354


Modules