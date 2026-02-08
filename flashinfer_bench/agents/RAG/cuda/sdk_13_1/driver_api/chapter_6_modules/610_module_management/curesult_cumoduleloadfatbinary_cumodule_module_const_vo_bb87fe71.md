# CUresult cuModuleLoadFatBinary (CUmodule *module, const void *fatCubin)

Load a module's data.

###### Parameters

**module**

  - Returned module
**fatCubin**

  - Fat binary to load

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,
CUDA_ERROR_INVALID_PTX, CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_NO_BINARY_FOR_GPU,
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
CUDA_ERROR_JIT_COMPILER_NOT_FOUND

###### Description

Takes a pointer fatCubin and loads the corresponding module module into the current context.
The pointer represents a fat binary object, which is a collection of different cubin and/or PTX files, all
representing the same device code, but compiled and optimized for different architectures.


CUDA Driver API TRM-06703-001 _vRelease Version  |  155


Modules


Prior to CUDA 4.0, there was no documented API for constructing and using fat binary objects by
programmers. Starting with CUDA 4.0, fat binary objects can be constructed by providing the -fatbin
option to nvcc. More information can be found in the nvcc document.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoad,
cuModuleLoadData, cuModuleLoadDataEx, cuModuleUnload