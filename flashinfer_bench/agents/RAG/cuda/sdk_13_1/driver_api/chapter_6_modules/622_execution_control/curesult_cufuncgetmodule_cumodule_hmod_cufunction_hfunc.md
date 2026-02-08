# CUresult cuFuncGetModule (CUmodule *hmod, CUfunction hfunc)

Returns a module handle.

###### Parameters

**hmod**

  - Returned module handle
**hfunc**

  - Function to retrieve module for

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_FOUND

###### Description

Returns in *hmod the handle of the module that function hfunc is located in. The lifetime of the
module corresponds to the lifetime of the context it was loaded in or until the module is explicitly
unloaded.

The CUDA runtime manages its own modules loaded into the primary context. If the handle returned
by this API refers to a module loaded by the CUDA runtime, calling cuModuleUnload() on that module
will result in undefined behavior.


Note:


Note that this function may also return error codes from previous, asynchronous launches.