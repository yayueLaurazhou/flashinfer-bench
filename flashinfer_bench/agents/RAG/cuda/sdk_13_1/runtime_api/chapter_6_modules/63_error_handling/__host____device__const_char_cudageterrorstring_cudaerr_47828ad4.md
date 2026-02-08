# __host____device__const char *cudaGetErrorString (cudaError_t error)

Returns the description string for an error code.

##### Parameters

**error**

  - Error code to convert to string

##### Returns

char* pointer to a NULL-terminated string


CUDA Runtime API vRelease Version  |  46


Modules

##### Description

Returns the description string for an error code. If the error code is not recognized, "unrecognized error
code" is returned.


See also:

cudaGetErrorName, cudaGetLastError, cudaPeekAtLastError, cudaError, cuGetErrorString