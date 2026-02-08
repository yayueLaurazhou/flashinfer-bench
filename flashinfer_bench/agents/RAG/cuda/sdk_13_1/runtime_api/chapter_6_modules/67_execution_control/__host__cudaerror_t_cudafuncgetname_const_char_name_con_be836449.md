# __host__cudaError_t cudaFuncGetName (const char **name, const void *func)

Returns the function name for a device entry function pointer.

##### Parameters

**name**

  - The returned name of the function
**func**

  - The function pointer to retrieve name for

##### Returns

cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidDeviceFunction

##### Description

Returns in **name the function name associated with the symbol func . The function name is
returned as a null-terminated string. This API may return a mangled name if the function is not


CUDA Runtime API vRelease Version  |  96


Modules


declared as having C linkage. If **name is NULL, cudaErrorInvalidValue is returned. If func is not
a device entry function, then it is assumed to be a cudaKernel_t and used as is.







cudaFuncGetName (C++ API)