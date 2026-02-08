# __host____device__const char *cudaGetErrorName (cudaError_t error)

Returns the string representation of an error code enum name.

##### Parameters

**error**

  - Error code to convert to string

##### Returns

char* pointer to a NULL-terminated string

##### Description

Returns a string containing the name of an error code in the enum. If the error code is not recognized,
"unrecognized error code" is returned.


See also:

cudaGetErrorString, cudaGetLastError, cudaPeekAtLastError, cudaError, cuGetErrorName