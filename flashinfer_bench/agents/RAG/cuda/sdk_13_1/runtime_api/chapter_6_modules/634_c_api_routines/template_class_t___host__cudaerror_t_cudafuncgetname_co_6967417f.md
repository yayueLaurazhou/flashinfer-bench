# template < class T > __host__cudaError_t cudaFuncGetName (const char **name, T *func)

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
declared as having C linkage. If **name is NULL, cudaErrorInvalidValue is returned. If func is not
a device entry function, cudaErrorInvalidDeviceFunction is returned.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.


CUDA Runtime API vRelease Version  |  465


Modules





cudaFuncGetName ( C API)