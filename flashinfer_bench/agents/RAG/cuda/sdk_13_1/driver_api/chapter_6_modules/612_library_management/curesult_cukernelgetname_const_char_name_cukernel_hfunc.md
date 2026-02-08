# CUresult cuKernelGetName (const char **name, CUkernel hfunc)

Returns the function name for a CUkernel handle.

###### Parameters

**name**

  - The returned name of the function
**hfunc**

  - The function handle to retrieve the name for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE

###### Description

Returns in **name the function name associated with the kernel handle hfunc . The function name is
returned as a null-terminated string. The returned name is only valid when the kernel handle is valid. If
the library is unloaded or reloaded, one must call the API again to get the updated name. This API may
return a mangled name if the function is not declared as having C linkage. If either **name or hfunc
is NULL, CUDA_ERROR_INVALID_VALUE is returned.


Note:


Note that this function may also return error codes from previous, asynchronous launches.