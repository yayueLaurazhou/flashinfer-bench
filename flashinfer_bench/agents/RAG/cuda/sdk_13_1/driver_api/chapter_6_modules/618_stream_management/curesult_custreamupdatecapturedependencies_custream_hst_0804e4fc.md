# CUresult cuStreamUpdateCaptureDependencies (CUstream hStream, CUgraphNode *dependencies, const CUgraphEdgeData *dependencyData, size_t numDependencies, unsigned int flags)

Update the set of dependencies in a capturing stream.

###### Parameters

**hStream**

  - The stream to update
**dependencies**

  - The set of dependencies to add


CUDA Driver API TRM-06703-001 _vRelease Version  |  351


Modules


**dependencyData**

  - Optional array of data associated with each dependency.
**numDependencies**

  - The size of the dependencies array
**flags**

  - See above

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_ILLEGAL_STATE

###### Description

Modifies the dependency set of a capturing stream. The dependency set is the set of nodes that the next
captured node in the stream will depend on along with the edge data for those dependencies.

Valid flags are CU_STREAM_ADD_CAPTURE_DEPENDENCIES and
CU_STREAM_SET_CAPTURE_DEPENDENCIES. These control whether the set
passed to the API is added to the existing set or replaces it. A flags value of 0 defaults to
CU_STREAM_ADD_CAPTURE_DEPENDENCIES.

Nodes that are removed from the dependency set via this API do not result in
CUDA_ERROR_STREAM_CAPTURE_UNJOINED if they are unreachable from the stream at
cuStreamEndCapture.

Returns CUDA_ERROR_ILLEGAL_STATE if the stream is not capturing.


See also:

cuStreamBeginCapture, cuStreamGetCaptureInfo