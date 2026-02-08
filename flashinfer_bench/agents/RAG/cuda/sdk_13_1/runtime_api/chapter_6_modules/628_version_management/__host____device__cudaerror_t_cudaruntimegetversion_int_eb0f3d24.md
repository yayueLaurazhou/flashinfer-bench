# __host____device__cudaError_t cudaRuntimeGetVersion (int *runtimeVersion)

Returns the CUDA Runtime version.

##### Parameters

**runtimeVersion**

  - Returns the CUDA Runtime version.

##### Returns

cudaSuccess, cudaErrorInvalidValue

##### Description

Returns in *runtimeVersion the version number of the current CUDA Runtime instance. The
version is returned as (1000 major + 10 minor). For example, CUDA 9.2 would be represented by
9020.

As of CUDA 12.0, this function no longer initializes CUDA. The purpose of this API is solely to return
a compile-time constant stating the CUDA Toolkit version in the above format.

This function automatically returns cudaErrorInvalidValue if the runtimeVersion argument is
NULL.





See also:

cudaDriverGetVersion, cuDriverGetVersion


CUDA Runtime API vRelease Version  |  316


Modules