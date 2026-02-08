# __host__cudaError_t cudaStreamEndCapture (cudaStream_t stream, cudaGraph_t *pGraph)

Ends capture on a stream, returning the captured graph.

##### Parameters

**stream**

  - Stream to query
**pGraph**

  - The captured graph

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorStreamCaptureWrongThread

##### Description

End capture on stream, returning the captured graph via pGraph. Capture must have been initiated
on stream via a call to cudaStreamBeginCapture. If capture was invalidated, due to a violation of the
rules of stream capture, then a NULL graph will be returned.

If the mode argument to cudaStreamBeginCapture was not cudaStreamCaptureModeRelaxed, this call
must be from the same thread as cudaStreamBeginCapture.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamCreate, cudaStreamBeginCapture, cudaStreamIsCapturing, cudaGraphDestroy