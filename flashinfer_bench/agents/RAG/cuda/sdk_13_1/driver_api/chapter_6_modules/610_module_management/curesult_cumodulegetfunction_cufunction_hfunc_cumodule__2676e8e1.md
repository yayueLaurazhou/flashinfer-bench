# CUresult cuModuleGetFunction (CUfunction *hfunc, CUmodule hmod, const char *name)

Returns a function handle.

###### Parameters

**hfunc**

  - Returned function handle
**hmod**

  - Module to retrieve function from
**name**

  - Name of function to retrieve

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in *hfunc the handle of the function of name name located in module hmod. If no function
of that name exists, cuModuleGetFunction() returns CUDA_ERROR_NOT_FOUND.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoad, cuModuleLoadData,
cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload