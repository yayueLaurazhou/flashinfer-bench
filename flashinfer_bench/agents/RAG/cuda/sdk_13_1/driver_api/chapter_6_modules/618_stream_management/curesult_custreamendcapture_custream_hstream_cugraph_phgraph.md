# CUresult cuStreamEndCapture (CUstream hStream, CUgraph *phGraph)

Ends capture on a stream, returning the captured graph.

###### Parameters

**hStream**

  - Stream to query
**phGraph**

  - The captured graph

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD

###### Description

End capture on hStream, returning the captured graph via phGraph. Capture must have been
initiated on hStream via a call to cuStreamBeginCapture. If capture was invalidated, due to a
violation of the rules of stream capture, then a NULL graph will be returned.

If the mode argument to cuStreamBeginCapture was not
CU_STREAM_CAPTURE_MODE_RELAXED, this call must be from the same thread as
cuStreamBeginCapture.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:


CUDA Driver API TRM-06703-001 _vRelease Version  |  340


Modules


cuStreamCreate, cuStreamBeginCapture, cuStreamIsCapturing, cuGraphDestroy