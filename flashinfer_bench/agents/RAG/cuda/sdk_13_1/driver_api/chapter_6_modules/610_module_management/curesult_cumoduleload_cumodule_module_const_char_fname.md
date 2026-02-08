# CUresult cuModuleLoad (CUmodule *module, const char *fname)

Loads a compute module.

###### Parameters

**module**

  - Returned module
**fname**

  - Filename of module to load

###### Returns

CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED,
CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE,


CUDA Driver API TRM-06703-001 _vRelease Version  |  152


Modules


CUDA_ERROR_INVALID_PTX, CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
CUDA_ERROR_NOT_FOUND, CUDA_ERROR_OUT_OF_MEMORY,
CUDA_ERROR_FILE_NOT_FOUND, CUDA_ERROR_NO_BINARY_FOR_GPU,
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
CUDA_ERROR_JIT_COMPILER_NOT_FOUND

###### Description

Takes a filename fname and loads the corresponding module module into the current context.
The CUDA driver API does not attempt to lazily allocate the resources needed by a module; if the
memory for functions and data (constant and global) needed by the module cannot be allocated,
cuModuleLoad() fails. The file should be a cubin file as output by nvcc, or a PTX file either as output
by nvcc or handwritten, or a fatbin file as output by nvcc from toolchain 4.0 or later, or a Tile IR file.


Note:


Note that this function may also return error codes from previous, asynchronous launches.


See also:

cuModuleGetFunction, cuModuleGetGlobal, cuModuleGetTexRef, cuModuleLoadData,
cuModuleLoadDataEx, cuModuleLoadFatBinary, cuModuleUnload