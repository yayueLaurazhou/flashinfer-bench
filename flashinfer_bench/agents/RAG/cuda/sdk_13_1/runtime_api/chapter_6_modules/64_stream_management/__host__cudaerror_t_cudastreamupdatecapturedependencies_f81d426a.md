# __host__cudaError_t cudaStreamUpdateCaptureDependencies (cudaStream_t stream, cudaGraphNode_t *dependencies, const cudaGraphEdgeData *dependencyData, size_t numDependencies, unsigned int flags)

Update the set of dependencies in a capturing stream.

##### Parameters

**stream**

  - The stream to update
**dependencies**

  - The set of dependencies to add
**dependencyData**

  - Optional array of data associated with each dependency.
**numDependencies**

  - The size of the dependencies array
**flags**

  - See above


CUDA Runtime API vRelease Version  |  69


Modules

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorIllegalState

##### Description

Modifies the dependency set of a capturing stream. The dependency set is the set of nodes that the next
captured node in the stream will depend on.

Valid flags are cudaStreamAddCaptureDependencies and cudaStreamSetCaptureDependencies. These
control whether the set passed to the API is added to the existing set or replaces it. A flags value of 0
defaults to cudaStreamAddCaptureDependencies.

Nodes that are removed from the dependency set via this API do not result in
cudaErrorStreamCaptureUnjoined if they are unreachable from the stream at cudaStreamEndCapture.

Returns cudaErrorIllegalState if the stream is not capturing.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cudaStreamBeginCapture, cudaStreamGetCaptureInfo,