# CUresult cuModuleUnload (CUmodule hmod)

Unloads a module.

###### Parameters

**hmod**

  - Module to unload

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_NOT_PERMITTED

###### Description

Unloads a module hmod from the current context. Attempting to unload a module which
was obtained from the Library Management API such as cuLibraryGetModule will return
CUDA_ERROR_NOT_PERMITTED.


Note:

**‣** Note that this function may also return error codes from previous, asynchronous launches.

**‣** Use of the handle after this call is undefined behavior.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoad,
cuModuleLoadData, cuModuleLoadDataEx, cuModuleLoadFatBinary


CUDA Driver API TRM-06703-001 _vRelease Version  |  156


Modules