# __host__cudaError_t cudaThreadExchangeStreamCaptureMode (cudaStreamCaptureMode *mode)

Swaps the stream capture interaction mode for a thread.

##### Parameters

**mode**

  - Pointer to mode value to swap with the current mode

##### Returns

cudaSuccess, cudaErrorInvalidValue


CUDA Runtime API vRelease Version  |  71


Modules

##### Description

Sets the calling thread's stream capture interaction mode to the value contained in *mode, and
overwrites *mode with the previous mode for the thread. To facilitate deterministic behavior across
function or module boundaries, callers are encouraged to use this API in a push-pop fashion:


During stream capture (see cudaStreamBeginCapture), some actions, such as a call to cudaMalloc,
may be unsafe. In the case of cudaMalloc, the operation is not enqueued asynchronously to a
stream, and is not observed by stream capture. Therefore, if the sequence of operations captured via
cudaStreamBeginCapture depended on the allocation being replayed whenever the graph is launched,
the captured graph would be invalid.

Therefore, stream capture places restrictions on API calls that can be made within or concurrently to a
cudaStreamBeginCapture-cudaStreamEndCapture sequence. This behavior can be controlled via this
API and flags to cudaStreamBeginCapture.

A thread's mode is one of the following:

This is the default mode. If the local thread has an

##### ‣ cudaStreamCaptureModeGlobal:

ongoing capture sequence that was not initiated with cudaStreamCaptureModeRelaxed at
cuStreamBeginCapture, or if any other thread has a concurrent capture sequence initiated
with cudaStreamCaptureModeGlobal, this thread is prohibited from potentially unsafe API
calls.
If the local thread has an ongoing capture

##### ‣ cudaStreamCaptureModeThreadLocal:

sequence not initiated with cudaStreamCaptureModeRelaxed, it is prohibited from
potentially unsafe API calls. Concurrent capture sequences in other threads are ignored.
The local thread is not prohibited from potentially

##### ‣ cudaStreamCaptureModeRelaxed:

unsafe API calls. Note that the thread is still prohibited from API calls which necessarily conflict
with stream capture, for example, attempting cudaEventQuery on an event that was last recorded
inside a capture sequence.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamBeginCapture


CUDA Runtime API vRelease Version  |  72


Modules